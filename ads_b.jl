"""
     ADS-B decoder implementation in Julia

Author: Luis Fernando Rodríguez Contreras
Date: January 2025

///// About this work
The present implementation follows the logic from the ModeS decoder Dump1090 (in C-language), using
the Adalm Pluto Software Defined Radio to obtain ADS-B samples from surrounding aircraft and print the
raw bytes, specially identifying the ICAO address

The ICAO address can then be placed in the search bar of an airplane tracker webpage like below
https://planefinder.net/

///// How to run
From the ADS_B_decoder.jl folder, execute: julia --startup-file=no --project=. ./ads_b.jl
"""

using LibIIO;                               # To access the standard IIO functions that the SDR uses
using FileIO;                               # To read data from a file that contains samples from the SDR
using Printf;                               # To print in a C-way (for debugging purposes)
using AdalmPluto;                           # To stablish connection with the SDR
using Plots;                                # To plot the ADS-B message wave

# Constants
const MODES_DEFAULT_RATE = 2000000;         # Sample frequency (2 MHz)
const MODES_DEFAULT_FREQ = 1090000000;      # Center frequency (1090 MHz)
const MODES_SAMPLES_COUNT::UInt = 131072;   # Number of (complex) samples that the Pluto input buffer should contain
const MODES_DATA_LEN::UInt = 262144;        # Number of real + imaginary samples obtained from the Pluto input buffer
const MODES_PREAMBLE_US::UInt = 8;          # Duration of the ModeS preamble in microseconds
const MODES_LONG_MSG_BITS::UInt = 112;      # Number of bits of data block of ADS-B packet
const MODES_LONG_MSG_BYTES::UInt = 14;      # Number of bytes of data block of ADS-B packet
const MODES_SHORT_MSG_BITS::UInt = 56;      # Number of bits of data block of other ModeS packets (not ADS-B)
const MODES_FULL_LEN::UInt = MODES_PREAMBLE_US + MODES_LONG_MSG_BITS;   # Complete ADS-B packet duration
const MODES_MAGN_LUT_SIZE::UInt = 33282;
const DATA_LEN::UInt = 262620;              # MODES_DATA_LEN + (MODES_FULL_LEN - 1) * 4. Size of Modes.data buffer

# Decoded message struct
mutable struct DecodedStruct
    msg::Array{Cuchar};                 # Binary message
    msgbits::Int;                       # Number of bits in message
    msgtype::Int;                       # Downlink format #
    crcok::Bool;                        # True if CRC was valid
    crc::UInt32;                        # Message CRC
    errorbit::Int;                      # Bit corrected. -1 if no bit corrected
    aa1::Int;                           # ICAO address byte 1
    aa2::Int;                           # ICAO address byte 2
    aa3::Int;                           # ICAO address byte 3
    phase_corrected::Bool;              # True if phase correction was applied
end

# Messages struct
struct MessageStruct
    data::Array{UInt8};                 # Raw IQ samples buffer
    maglut::Array{UInt16};              # I/Q -> Magnitude lookup table (LUT)
    magnitude::Array{UInt16};           # Magnitude vector
end

# Pluto struct
mutable struct PlutoStruct
    ctx::Ptr{iio_context};              # Pluto device context
    dev::Ptr{iio_device};               # Pointer to IIO device
    rx0_i::Ptr{iio_channel};            # Pointer to the I channel in Rx
    rx0_q::Ptr{iio_channel};            # Pointer to the Q channel in Rx
    rxbuf::Ptr{iio_buffer};             # Non-cycling RX buffer
    i_raw_samples::Array{UInt8};        # Temporary storage of raw I samples
    q_raw_samples::Array{UInt8};        # Temporary storage of raw Q samples        
end

"""
Initialization function
Populate the I/Q -> Magnitude lookup table. Computationally speaking, using the LUT is faster
than performing the sqrt and round functions on the fly
We scale to 0-255 range multiplying by 1.4 in order to ensure that every different
I/Q pair will result in a different magnitude value, not losing any resolution
"""
function populateLUT(Modes::MessageStruct)
    for i in 0:128
        for q in 0:128
            Modes.maglut[(i * 129 + q) + 1] = round(sqrt(i*i + q*q) * 360);
        end
    end
end

"""
PlutoSDR handling, to establish the connection with the SDR and define center frequency, sampling
frequency, gain control mode, and initialize the RX channels, among others
"""
function initPluto(Pluto::PlutoStruct)
    println("[ADS-B Decoder] Acquiring IIO context");
    Pluto.ctx = C_iio_create_network_context("pluto.local")
    if(C_iio_context_get_devices_count(Pluto.ctx) == 0) error("No Adalm Pluto device found"); end

    println("[ADS-B Decoder] Acquiring AD9361 streaming devices");
    Pluto.dev = C_iio_context_find_device(Pluto.ctx, "cf-ad9361-lpc");
    if(Pluto.dev == C_NULL) error("Error opening the PLUTOSDR device"); end

    println("[ADS-B Decoder] Acquiring AD9361 phy channel 0");
    phyDevice = C_iio_context_find_device(Pluto.ctx, "ad9361-phy");
    phyChannel = C_iio_device_find_channel(phyDevice, "voltage0", false);

    C_iio_channel_attr_write(phyChannel, "rf_port_select", "A_BALANCED");
    C_iio_channel_attr_write_longlong(phyChannel, "rf_bandwidth", MODES_DEFAULT_RATE);
    C_iio_channel_attr_write_longlong(phyChannel, "sampling_frequency", MODES_DEFAULT_RATE);

    loChannel = C_iio_device_find_channel(phyDevice, "altvoltage0", true);
    C_iio_channel_attr_write_longlong(loChannel, "frequency", MODES_DEFAULT_FREQ);

    println("[ADS-B Decoder] Initializing AD9361 IIO streaming channels");
    Pluto.rx0_i = C_iio_device_find_channel(Pluto.dev, "voltage0", false);
    if(Pluto.rx0_i == C_NULL) Pluto.rx0_i = C_iio_device_find_channel(Pluto.dev, "altvoltage0", false); end
    Pluto.rx0_q = C_iio_device_find_channel(Pluto.dev, "voltage1", false);
    if(Pluto.rx0_q == C_NULL) Pluto.rx0_q = C_iio_device_find_channel(Pluto.dev, "altvoltage1", false); end

    ad9361_baseband_auto_rate(phyDevice, MODES_DEFAULT_RATE);

    println("[ADS-B Decoder] Enabling IIO streaming channels");
    C_iio_channel_enable(Pluto.rx0_i);
    C_iio_channel_enable(Pluto.rx0_q);

    println("[ADS-B Decoder] Creating non-cyclic IIO buffers");
    Pluto.rxbuf = C_iio_device_create_buffer(Pluto.dev, MODES_SAMPLES_COUNT, false);
    if(Pluto.rxbuf == C_NULL) error("Could not create RX buffer"); end

    C_iio_channel_attr_write(phyChannel, "gain_control_mode", "slow_attack");

    println("[ADS-B Decoder] Completed AdalmPluto config");
end

"""
Compute the Magnitude Vector
Turn I/Q samples stored in Modes.data into the magnitude vector stored into Modes.magnitude
This is done since the Mode S downlink modulation scheme is ASK, so for this simple scheme the
signal of interest comes in the form of the magnitude of the I/Q signals
"""
function computeMagnitudeVector(Modes::MessageStruct)
    i::Int = 0;
    q::Int = 0;
    
    for j in 1:2:DATA_LEN
        # Compute the magnitude vector. It's just SQRT(I^2 + Q^2), but
        # we rescale to the 0-255 range to exploit the full resolution.
        i = Modes.data[j] - 127;
        q = Modes.data[j + 1] - 127;

        if(i < 0) i = -i; end
        if(q < 0) q = -q; end

        Modes.magnitude[Int((j+1) / 2)] = Modes.maglut[(i * 129 + q) + 1];
    end
end

"""
Parity table for MODE S Messages.
The table contains 112 elements, every element corresponds to a bit set
in the message, starting from the first bit of actual data after the
preamble.

For messages of 112 bit, the whole table is used.
For messages of 56 bits only the last 56 elements are used.

The algorithm is as simple as xoring all the elements in this table
for which the corresponding bit on the message is set to 1.

The latest 24 elements in this table are set to 0 as the checksum at the
end of the message should not affect the computation.

Note: this function can be used with DF11 and DF17, other modes have
the CRC xored with the sender address as they are reply to interrogations,
but a casual listener can't split the address from the checksum.
"""
modes_checksum_table = UInt32[
    0x3935ea, 0x1c9af5, 0xf1b77e, 0x78dbbf, 0xc397db, 0x9e31e9, 0xb0e2f0, 0x587178,
	0x2c38bc, 0x161c5e, 0x0b0e2f, 0xfa7d13, 0x82c48d, 0xbe9842, 0x5f4c21, 0xd05c14,
	0x682e0a, 0x341705, 0xe5f186, 0x72f8c3, 0xc68665, 0x9cb936, 0x4e5c9b, 0xd8d449,
	0x939020, 0x49c810, 0x24e408, 0x127204, 0x093902, 0x049c81, 0xfdb444, 0x7eda22,
	0x3f6d11, 0xe04c8c, 0x702646, 0x381323, 0xe3f395, 0x8e03ce, 0x4701e7, 0xdc7af7,
	0x91c77f, 0xb719bb, 0xa476d9, 0xadc168, 0x56e0b4, 0x2b705a, 0x15b82d, 0xf52612,
	0x7a9309, 0xc2b380, 0x6159c0, 0x30ace0, 0x185670, 0x0c2b38, 0x06159c, 0x030ace,
	0x018567, 0xff38b7, 0x80665f, 0xbfc92b, 0xa01e91, 0xaff54c, 0x57faa6, 0x2bfd53,
	0xea04ad, 0x8af852, 0x457c29, 0xdd4410, 0x6ea208, 0x375104, 0x1ba882, 0x0dd441,
	0xf91024, 0x7c8812, 0x3e4409, 0xe0d800, 0x706c00, 0x383600, 0x1c1b00, 0x0e0d80,
	0x0706c0, 0x038360, 0x01c1b0, 0x00e0d8, 0x00706c, 0x003836, 0x001c1b, 0xfff409,
	0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000,
	0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000,
	0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000
];

"""
Get the 24-bit checksum
"""
function modesChecksum(msg::Vector{UInt8}, bits::Int)
    crc::UInt32 = 0;
    offset::Int = 0;
    byte::Int = 0;
    bit::Int = 0;
    bitmask::Int = 0;

    if (bits == 112)
        offset = 0;
    else
        offset = 56;
    end
    
    for iterator in 1:bits
        byte = Int(ceil(iterator/8));           # Current byte number (starts at 1 because of Julia indexing)
        bit = (iterator - 1) % 8;               # Current bit inside the current byte
        bitmask = 1 << (7 - bit);

        # If bit is set, xor with corresponding table entry
        if ((msg[byte] & bitmask) != 0)
            crc = xor(crc, modes_checksum_table[iterator + offset]);
        end
    end

    return crc;                                 # 24 bit checksum
end

"""
Given the Downlink Format (DF) of the message, return the message length in bits
"""
function modesMessageLenByType(type::Int)
    if (type == 16 || type == 17 || type == 19 || type == 20 || type == 21)
        return MODES_LONG_MSG_BITS;
    else
        return MODES_SHORT_MSG_BITS;
    end
end

"""
Try to fix single bit errors using the checksum. On success modifies the original buffer
with the fixed version, and returns the position of the error bit. Otherwise if fixing
failed -1 is returned
"""
function fixSingleBitErrors(msg::Vector{UInt8}, bits::Int)
    aux  = zeros(UInt8, UInt(MODES_LONG_MSG_BITS / 8));
    crc1::UInt32 = 0;
    crc2::UInt32 = 0;
    byte::Int = 0;
    bit::Int = 0;
    bitmask::Int = 0;

    for iterator in 1:bits
        byte = Int(ceil(iterator/8));           # Current byte number (starts at 1 bc of Julia indexing)
        bit = (iterator - 1) % 8;               # Current bit inside the current byte
        bitmask = 1 << (7 - bit);

        for index in 1:(Int(bits/8))            # Copy bytes of msg into aux
            aux[index] = msg[index];
        end

        aux[byte] = xor(aux[byte], bitmask);    # Flip j-th bit

        crc1 = (UInt32(aux[Int((bits / 8) - 2)]) << 16 |
              UInt32(aux[Int((bits / 8) - 1)]) << 8  |
              UInt32(aux[Int((bits / 8))]));
        crc2 = modesChecksum(aux, bits);

        if (crc1 == crc2)
            # The error is fixed. Overwrite the original buffer with the corrected sequence,
            # and return the error bit position
            for index in 1:(Int(bits/8))        # Copy bytes of aux into msg
                msg[index] = aux[index];
            end

            return iterator;
        end
    end

    return -1;
end

"""
This function does not really correct the phase of the message, it just
applies a transformation to the first sample representing a given bit:

If the previous bit was one, we amplify it a bit.
If the previous bit was zero, we decrease it a bit.

This simple transformation makes the message a bit more likely to be
correctly decoded for out of phase messages:

When messages are out of phase there is more uncertainty in
sequences of the same bit multiple times, since 11111 will be
transmitted as continuously altering magnitude (high, low, high, low...)

However because the message is out of phase some part of the high
is mixed in the low part, so that it is hard to distinguish if it is
a zero or a one.

However when the message is out of phase passing from 0 to 1 or from
1 to 0 happens in a very recognizable way, for instance in the 0 -> 1
transition, magnitude goes low, high, high, low, and one of the
two middle samples the high will be *very* high as part of the previous
or next high signal will be mixed there.

Applying our simple transformation we make more likely if the current
bit is a zero, to detect another zero. Symmetrically if it is a one
it will be more likely to detect a one because of the transformation.
In this way similar levels will be interpreted more likely in the
correct way
"""
function applyPhaseCorrection(m::Array{UInt16}, idx::Int)
    idx += 16;    # Skip preamble

    for iterator in 0:2:(((MODES_LONG_MSG_BITS - 1) * 2) - 1)
        if (m[idx + iterator] > m[idx + iterator + 1])
            m[idx + iterator + 2] = trunc(UInt16, (m[idx + iterator + 2] * 5) / 4);    # One
        else
            m[idx + iterator + 2] = trunc(UInt16, (m[idx + iterator + 2] * 4) / 5);    # Zero
        end
    end
end

"""
Return true if the message is out of phase
Return false if the message is not particularly out of phase

Note: this function will access m[-1], so the caller should make sure to call it only
if we are not at the start of the current buffer.
"""
function detectOutOfPhase(m::Array{UInt16}, idx::Int)
    if (m[idx + 3] > m[idx + 2] / 3) return true end;
    if (m[idx + 10] > m[idx + 9] / 3) return true end;
    if (m[idx + 6] > m[idx + 7] / 3) return true end;
    if (m[idx - 1] > m[idx + 1] / 3) return true end;
    return false;
end

"""
Decode a raw Mode S message demodulated as a stream of bytes by detectModeS()
"""
function decodeModesMessage(mm::DecodedStruct, msg::Vector{UInt8})
    crc2::UInt32 = 0;   # Computed CRC, used to verify the message CRC

    # Work on our local copy
    for iterator in 1:MODES_LONG_MSG_BYTES
        mm.msg[iterator] = msg[iterator];
    end

    # Get the Downlink Format (DF) and message length in bits
    mm.msgtype = mm.msg[1] >> 3;
    mm.msgbits = modesMessageLenByType(mm.msgtype);

    # CRC is always the last three bytes
    mm.crc = (UInt32(mm.msg[Int((mm.msgbits / 8) - 2)]) << 16 |
              UInt32(mm.msg[Int((mm.msgbits / 8) - 1)]) << 8  |
              UInt32(mm.msg[Int((mm.msgbits / 8))]));
    crc2 = modesChecksum(mm.msg, mm.msgbits);

    # Check CRC and fix single bit errors using the CRC when possible (DF 17)
    mm.errorbit = -1;   # No error
    mm.crcok = (mm.crc == crc2);

    if (!(mm.crcok) && (mm.msgtype == 17))
        if ((mm.errorbit = fixSingleBitErrors(mm.msg, mm.msgbits)) != -1)
            mm.crc = modesChecksum(mm.msg, mm.msgbits);
            mm.crcok = true;
        end
    end

    # ICAO address
    mm.aa1 = mm.msg[2];
    mm.aa2 = mm.msg[3];
    mm.aa3 = mm.msg[4];
end

"""
This function gets a decoded Mode S Message (ADS-B) and prints it on the screen in a human-readable
format. Additionally, it saves plots of the decoded ADS-B packet plus some samples before and
after, such that it is recognizable in a visual way. The preamble is also plotted separately
"""
function displayModesMessage(mm::DecodedStruct, adsb_index::Int)
    
    # ADS-B has DF = 17
    if (mm.msgtype == 17)
        println("--> Raw bytes INIT <--");
        # Show the raw message
        for iterator in 1:(Int(mm.msgbits/8))
            @printf("%02x", mm.msg[iterator]);
        end
        println("\n--> Raw bytes END <--");

        if (mm.errorbit != -1)
            println("Single bit error fixed, bit ", mm.errorbit);
        end
        
        println("DF 17: ADS-B message");
        @printf("  ICAO Address   : %02x%02x%02x\n", mm.aa1, mm.aa2, mm.aa3);

        # Create different plots to show the ADS-B message signal
        x = range(1, 2240);     # 240 samples of complete ADS-B + 1000 before and after
        y = zeros(UInt16, 2240);
        for iterator in 1:2240
            y[iterator] = Modes.magnitude[adsb_index + iterator - 1 - 1000];
        end
        p = plot(x, y);

        # Save plot as a .png file
        savefig(p, "plot_2240_samples.png")

        x = range(1, 1240);     # 240 samples of complete ADS-B + 500 before and after
        y = zeros(UInt16, 1240);
        for iterator in 1:1240
            y[iterator] = Modes.magnitude[adsb_index + iterator - 1 - 500];
        end
        p = plot(x, y);

        # Save plot as a .png file
        savefig(p, "plot_1240_samples.png")

        x = range(1, 440);      # 240 samples of complete ADS-B + 100 before and after
        y = zeros(UInt16, 440);
        for iterator in 1:440
            y[iterator] = Modes.magnitude[adsb_index + iterator - 1 - 100];
        end
        p = plot(x, y);

        # Save plot as a .png file
        savefig(p, "plot_ads-b.png")

        x = range(1, 16);      # 16 samples of preamble
        y = zeros(UInt16, 16);
        for iterator in 1:16
            y[iterator] = Modes.magnitude[adsb_index + iterator - 1];
        end
        p = plot(x, y);

        # Save plot as a .png file
        savefig(p, "plot_preamble.png")

        println("Press Enter to continue");
        readline()  # Waits for the user to press the Enter key
    end
end

"""
Detect a Mode S message inside the magnitude buffer and of size 'len' bytes.
Every detected Mode S message is converted into a stream of bits and passed to
the function to display it
"""
function detectModeS(Modes::MessageStruct, mlen::UInt32)
    # Create a byte buffer to store the data
    bits = zeros(Cuchar, MODES_LONG_MSG_BITS);
    msg  = zeros(Cuchar, UInt(MODES_LONG_MSG_BITS / 2));
    aux  = zeros(UInt16, UInt(MODES_LONG_MSG_BITS * 2));
    decoded = DecodedStruct(zeros(Cuchar, MODES_LONG_MSG_BYTES), 0, 0, false, 0, 0, 0, 0, 0, false);
    use_correction = false;
    good_message = false;
    errors::Int = 0;
    high::Int = 0;
    low::Int = 0;
    delta::Int = 0;
    temp::Int = 0;
    j::Int = 1;
    adsb_index::Int = 0;

    """
    The Mode S preamble is made of impulses of 0.5 microseconds at
	 * the following time offsets:
	 *
	 * 0   - 0.5 usec: first impulse.
	 * 1.0 - 1.5 usec: second impulse.
	 * 3.5 - 4   usec: third impulse.
	 * 4.5 - 5   usec: last impulse.
	 *
	 * Since we are sampling at 2 Mhz every sample in our magnitude vector
	 * is 0.5 usec, so the preamble will look like this, assuming there is
	 * an impulse at offset 0 in the array:
	 *
	 * 0   -----------------
	 * 1   -
	 * 2   ------------------
	 * 3   --
	 * 4   -
	 * 5   --
	 * 6   -
	 * 7   ------------------
	 * 8   --
	 * 9   -------------------
    """

    while (j <= (mlen - MODES_FULL_LEN * 2))
        if (!use_correction)        # If we have not checked yet
            """
            First check of relations between the first 10 samples representing a valid preamble.
            We don't even investigate further if this simple test is not passed.
            """
            if (!(Modes.magnitude[j] > Modes.magnitude[j + 1] &&
                Modes.magnitude[j + 1] < Modes.magnitude[j + 2] &&
                Modes.magnitude[j + 2] > Modes.magnitude[j + 3] &&
                Modes.magnitude[j + 3] < Modes.magnitude[j] &&
                Modes.magnitude[j + 4] < Modes.magnitude[j] &&
                Modes.magnitude[j + 5] < Modes.magnitude[j] &&
                Modes.magnitude[j + 6] < Modes.magnitude[j] &&
                Modes.magnitude[j + 7] > Modes.magnitude[j + 8] &&
                Modes.magnitude[j + 8] < Modes.magnitude[j + 9] &&
                Modes.magnitude[j + 9] > Modes.magnitude[j + 6]))
                j = j + 1;
                continue;
            end

            """
            The samples between the two spikes must be < than the average of the high
            spikes level. We don't test bits too near to the high levels as signals can
            be out of phase so part of the energy can be in the near samples.
            """
            # 'trunc' takes the decimal portion away and gives back the specified data type
            high = trunc(Int, ((Modes.magnitude[j] + Modes.magnitude[j+2] + Modes.magnitude[j+7] + Modes.magnitude[j+9]) / 6));
            
            if (Modes.magnitude[j + 4] >= high || Modes.magnitude[j + 5] >= high)
                j = j + 1;
                continue;
            end

            """
            Similarly samples in the range 11-14 must be low, as it is the space between
            the preamble and real data. Again we don't test bits too near to high levels, see above
            """
            if (Modes.magnitude[j + 11] >= high || Modes.magnitude[j + 12] >= high ||
                Modes.magnitude[j + 13] >= high || Modes.magnitude[j + 14] >= high)
                j = j + 1;
                continue;
            end
        end

        # If the previous attempt with this message failed, retry using magnitude correction
        if (use_correction)
            
            for index in 1:(length(aux))
                aux[index] = Modes.magnitude[index + j + MODES_PREAMBLE_US * 2];
            end
            
            if (detectOutOfPhase(Modes.magnitude, j))
                applyPhaseCorrection(Modes.magnitude, j);
            end
        end

        # Decode all the next 112 bits, regardless of the actual message
        # size. We'll check the actual message type later
        # The value assignment follows the Manchester code: logic 1 is falling edge \ and logic 0
        # is rising edge /
        errors = 0;

        for iterator in 0:2:(MODES_LONG_MSG_BITS * 2 - 1)
            low  = Modes.magnitude[j + iterator + MODES_PREAMBLE_US * 2];
            high = Modes.magnitude[j + iterator + MODES_PREAMBLE_US * 2 + 1];
            delta = low - high;

            if (delta < 0) delta = -delta; end
            
            # Note that "iterator" increments 2 units each loop, so it is divided by 2 when used
            # inside the "bits" array such that it is populated element by element
            if (iterator > 0 && delta < 256)
                bits[Int(iterator / 2) + 1] = bits[Int(iterator / 2)];
            elseif (low == high)
                # If two adjacent samples have same magnitude, it was noise detected as a valid preamble
                bits[Int(iterator / 2) + 1] = 2;   # error
                if (iterator < MODES_SHORT_MSG_BITS * 2) errors += 1; end
            elseif (low > high)
                bits[Int(iterator / 2) + 1] = 1;
            else   #low < high
                bits[Int(iterator / 2) + 1] = 0;
            end
        end

        # Restore the original message if we used magnitude correction
        if (use_correction)
            for index in 1:(length(aux))
                Modes.magnitude[index + j + MODES_PREAMBLE_US * 2] = aux[index];
            end
        end

        # Pack bits into bytes
        for iterator in 1:8:MODES_LONG_MSG_BITS
            msg[trunc(Int, (iterator+7) / 8)] = 
                bits[iterator] << 7 |
                bits[iterator + 1] << 6 |
                bits[iterator + 2] << 5 |
                bits[iterator + 3] << 4 |
                bits[iterator + 4] << 3 |
                bits[iterator + 5] << 2 |
                bits[iterator + 6] << 1 |
                bits[iterator + 7];
        end

        # "msgtype" is just the Downlink Format (decimal 17 for ADS-B)
        msgtype::Int = msg[1] >> 3;
        # "msglen" is the size in bytes of the message, depending on the previous Downlink Format
        msglen::Int = Int(modesMessageLenByType(msgtype) / 8);

        # Last check, high and low bits are different enough in magnitude to mark this
        # as real message and not just noise?
        delta = 0;
        low = 0;
        high = 0;
        temp = 0;
        for iterator in 0:2:(msglen * 8 * 2 - 1)
            # luisfrod: Julia --> delta += abs(low-high) overflows when done in 1 line!
            low = Modes.magnitude[j + iterator + MODES_PREAMBLE_US * 2];
            high = Modes.magnitude[j + iterator + MODES_PREAMBLE_US * 2 + 1];
            temp = low - high;
            if (temp < 0) temp = -temp; end
            delta += temp;
        end
        # "delta" below is some kind of average of all the differences between
        # current and next samples (low and high)
        delta = trunc(Int, delta / (msglen * 4));

        # Filter for an average delta of three is small enough to let almost every kind of
        # message to pass, but high enough to filter some random noise
        if (delta < 10 * 255)
            use_correction = false;
            j = j + 1;
            continue;
        end

        # If we reached this point, and error is zero, we are very likely with a Mode S message
        # in our hands, but it may still be broken and CRC may not be correct. This is handled by
        # the next layer
        if (errors == 0)
            # Save the index of the potential ADS-B message
            adsb_index = j;

            # Decode the received message
            decodeModesMessage(decoded, msg);

            # Skip this message if we are sure it's fine
            if (decoded.crcok)
                j += (MODES_PREAMBLE_US + (msglen * 8)) * 2;
                good_message = true;
            end

            # Pass data to the next layer
            useModesMessage(decoded, adsb_index);
        end

        # Retry with phase correction if possible
        if (!good_message && !use_correction)
            j = j - 1;
            use_correction = true;
        else
            use_correction = false;
        end

        # Increment j before next iteration
        j = j + 1;
    end

end

"""
When a new message is available, because it was decoded from the Pluto device or read from a file,
or any other way we can receive a decoded message, we call this function in order to use the message.
Basically this function passes a raw message to the upper layers for further processing and visualization
"""
function useModesMessage(mm::DecodedStruct, adsb_index::Int)
    if (mm.crcok)
        displayModesMessage(mm, adsb_index);
    end
end

"""
Pluto Callback function to populate the data buffer
"""
function callbackPluto(buf::Array{UInt8}, len::UInt32)
    i::UInt32 = 1;
    while (i <= len)
        buf[i] = xor(buf[i], 0x80); 
        i = i+1;
    end

    if (len > MODES_DATA_LEN)
        len = MODES_DATA_LEN;
    end

    # Move the last part of the previous buffer, that was not processed,
    # on the start of the new buffer
    for index in 1:((MODES_FULL_LEN-1)*4)
        Modes.data[index] = Modes.data[index + (length(Modes.data) - ((MODES_FULL_LEN-1)*4))];
    end

    # Read the new data
    for index in 1:(len)
        Modes.data[index + (MODES_FULL_LEN-1)*4] = buf[index];
    end
end

"""
Asynchronous data reader
"""
function asyncReader(Pluto::PlutoStruct)    
    # Create a byte buffer to store the data
    cb_buf = zeros(Cuchar, 262144);                 # MODES_DATA_LEN = 262144

    index = 1                                       # "j" in dump1090
    C_iio_buffer_refill(Pluto.rxbuf);               # 524288 bytes refilled in Pluto's input buffer
    p_inc = C_iio_buffer_step(Pluto.rxbuf);         # p_inc = 4 bytes, size of a complex sample from Pluto
    p_end = C_iio_buffer_end(Pluto.rxbuf);
    p_dat = C_iio_buffer_first(Pluto.rxbuf, Pluto.rx0_i);
    
    while (p_dat < p_end)
        i = unsafe_load(Ptr{Int16}(p_dat), 1);      # Extract the Real (I) portion of the sample
        q = unsafe_load(Ptr{Int16}(p_dat), 2);      # Extract the Imag (Q) portion of the sample
        i_twosComp = (2^16 + i) % 2^16;             # 2's complement of I in case i is negative
        q_twosComp = (2^16 + q) % 2^16;             # 2's complement of Q in case q is negative
        cb_buf[index] = (i_twosComp>>4) & 0xFF;     # 4-bit right shift and keep lower 8 bits
        cb_buf[index+1] = (q_twosComp>>4) & 0xFF;   # 4-bit right shift and keep lower 8 bits

        # Done with the current pair of I/Q samples, go to next sample pair
        index = index+2;
        p_dat += p_inc;
    end

    callbackPluto(cb_buf, UInt32(MODES_DATA_LEN));
end

"""
Read data from file instead of using the Adalm Pluto
"""
function readDataFromFile(read_byte::Int)
    # Move the last part of the previous buffer, that was not processed,
    # on the start of the new buffer
    for index in 1:((MODES_FULL_LEN-1)*4)
        Modes.data[index] = Modes.data[index + (length(Modes.data) - ((MODES_FULL_LEN-1)*4))];
    end

    # Data file memory offset
    offset = read_byte * MODES_DATA_LEN;

    # Read the data
    file = open("pluto_out", "r");          # pluto_out      -> small file with 1 ADS-B message
    #file = open("long_recording", "r");     # long_recording -> big file with 6 ADS-B messages
    buf = read(file);
    close(file);

    for index in 1:MODES_DATA_LEN
        Modes.data[index + (MODES_FULL_LEN-1)*4] = buf[index + offset];
    end
end

# Create an instance of the Pluto data type
Pluto = PlutoStruct(C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
        zeros(UInt8, (DATA_LEN) ÷ 2),
        zeros(UInt8, (DATA_LEN) ÷ 2));

# Initialize the Message struct
Modes = MessageStruct(zeros(UInt8, DATA_LEN),
        zeros(UInt16, (129*129)),       # The Magnitude LUT is 129x129 between I and Q values
        zeros(UInt16, DATA_LEN));

# Initialize the raw IQ samples buffer to a known value
for iterator in 1:DATA_LEN
    Modes.data[iterator] = 127;
end

# Populate Magnitude LUT
populateLUT(Modes);

# Variable that keeps the number of iterations reading from a file
read_byte::Int = 0;

# Initialize the Pluto SDR
initPluto(Pluto);

while(true)
##while(read_byte <= 2)                   # For pluto_out, read_byte = [0,2] | for long_recording, read_byte = [0,9]
    # Read from Pluto (asyncReader) or read from a file
    asyncReader(Pluto);                 # luisfrod: to read from Adalm Pluto
    ##
    ##readDataFromFile(read_byte);       # luisfrod: to read from file
    ##global read_byte += 1;
    ##

    # Now that the data is ready, process it
    computeMagnitudeVector(Modes);

    # Process data
    detectModeS(Modes, UInt32(DATA_LEN/2));
end
