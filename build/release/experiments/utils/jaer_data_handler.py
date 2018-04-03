# taken from liquid_state_machine repo

import struct
import os
import pandas as pd

V3 = "aedat3"
V2 = "aedat"  # current 32bit file format
V1 = "dat"  # old format

EVT_DVS = 0  # DVS event type
EVT_APS = 1  # APS event


def load_rosbag():
    pass


def load_jaer(datafile='/tmp/aerout.dat', length=0, version=V2, debug=1, camera='DVS128'):
    """
    load AER data file and parse these properties of AE events:
    - timestamps (in us),
    - x,y-position [0..127]
    - polarity (0/1)

    @param datafile - path to the file to read
    @param length - how many bytes(B) should be read; default 0=whole file
    @param version - which file format version is used: "aedat" = v2, "dat" = v1 (old)
    @param debug - 0 = silent, 1 (default) = print summary, >=2 = print all debug
    @param camera='DVS128' or 'DAVIS240'
    @return (ts, xpos, ypos, pol) 4-tuple of lists containing data of all events;
    """
    # constants
    aeLen = 8  # 1 AE event takes 8 bytes
    readMode = '>II'  # struct.unpack(), 2x ulong, 4B+4B
    td = 0.000001  # timestep is 1us
    if (camera == 'DVS128'):
        xmask = 0x00fe
        xshift = 1
        ymask = 0x7f00
        yshift = 8
        pmask = 0x1
        pshift = 0
    elif (camera == 'DAVIS240'):  # values take from scripts/matlab/getDVS*.m
        xmask = 0x003ff000
        xshift = 12
        ymask = 0x7fc00000
        yshift = 22
        pmask = 0x800
        pshift = 11
        eventtypeshift = 31
    else:
        raise ValueError("Unsupported camera: %s" % (camera))

    if (version == V1):
        print ("using the old .dat format")
        aeLen = 6
        readMode = '>HI'  # ushot, ulong = 2B+4B

    aerdatafh = open(datafile, 'rb')
    k = 0  # line number
    p = 0  # pointer, position on bytes
    statinfo = os.stat(datafile)
    if length == 0:
        length = statinfo.st_size

    # header
    lt = aerdatafh.readline()
    while lt and lt[0] == '#':
        p += len(lt)
        k += 1
        lt = aerdatafh.readline()
        if debug >= 2:
            print (str(lt))
        continue

    # variables to parse
    timestamps = []
    xaddr = []
    yaddr = []
    pol = []

    # read data-part of file
    aerdatafh.seek(p)
    s = aerdatafh.read(aeLen)
    p += aeLen

    # print (xmask, xshift, ymask, yshift, pmask, pshift)
    while p < length:
        addr, ts = struct.unpack(readMode, s)
        # parse event type
        if (camera == 'DAVIS240'):
            eventtype = (addr >> eventtypeshift)
        else:  # DVS128
            eventtype = EVT_DVS

        # parse event's data
        if (eventtype == EVT_DVS):  # this is a DVS event
            x_addr = (addr & xmask) >> xshift
            y_addr = (addr & ymask) >> yshift
            a_pol = (addr & pmask) >> pshift

            if debug >= 3:
                print("ts->", ts)  # ok
                print("x-> ", x_addr)
                print("y-> ", y_addr)
                print("pol->", a_pol)

            timestamps.append(ts)
            xaddr.append(x_addr)
            yaddr.append(y_addr)
            pol.append(a_pol)

        aerdatafh.seek(p)
        s = aerdatafh.read(aeLen)
        p += aeLen

    if debug > 0:
        try:
            print ("read %i (~ %.2fM) AE events, duration= %.2fs" % (
                len(timestamps), len(timestamps) / float(10 ** 6), (timestamps[-1] - timestamps[0]) * td))
            n = 5
            print ("showing first %i:" % (n))
            print ("timestamps: %s \nX-addr: %s\nY-addr: %s\npolarity: %s" % (
                timestamps[0:n], xaddr[0:n], yaddr[0:n], pol[0:n]))
        except:
            print ("failed to print statistics")

    return timestamps, xaddr, yaddr, pol


def load_aedat31(datafile='/tmp/aerout.dat', length=0, debug=1):
    """
    load AER data file (in format AEDAT 3.1) and parse these properties of AE events:
    - timestamps (in us),
    - x,y-position [0..127]
    - polarity (0/1)

    @param datafile - path to the file to read
    @param debug - 0 = silent, 1 (default) = print summary, >=2 = print all debug
    @return (ts, xpos, ypos, pol) 4-tuple of lists containing data of all events;
    """
    # ATTENTION: this implementation assumes that the aedat file contains only polarity events,
    # and does not provide a full implementation of the AEDAT 3.1 specification!

    # constants
    header_len = 2 * 2 + 6 * 4  # bytes
    event_len = 4 * 2  # bytes
    header_format = '<HHIIIIII'  # cp. struct.unpack and aedat header definition
    event_format = '<II'
    header_format = '<hhiiiiii'
    event_format = '<ii'

    td = 0.000001  # timestep is 1us

    aerdatafh = open(datafile, 'rb')
    k = 0  # line number
    p = 0  # pointer, position on bytes
    statinfo = os.stat(datafile)
    if length == 0:
        length = statinfo.st_size
    # print ("file size", length)

    # File header
    lt = aerdatafh.readline()
    while lt and lt[0] == '#':
        p += len(lt)
        k += 1
        if debug >= 2:
            print('header: ', str(lt))
        lt = aerdatafh.readline()
        continue

    # variables to parse
    timestamps = []
    xaddr = []
    yaddr = []
    pol = []

    blocks_read = 0
    while True:
        aerdatafh.seek(p)
        s = aerdatafh.read(header_len)
        p += header_len

        if len(s) == 0:
            break

        event_type, event_source, event_size, event_ts_offset, \
        event_ts_overflow, event_capacity, event_number, event_valid = struct.unpack(header_format, s)
        blocks_read += 1

        # print(event_number)
        # an event block contains event_number events.
        for event_in_block in range(event_number):
            # print(event_in_block)
            aerdatafh.seek(p)
            s = aerdatafh.read(event_len)  # Block-wise reading could enhance performance significantly
            p += event_len
            data, timestamp = struct.unpack(event_format, s)

            x = (data >> 17) & 0x00001FFF
            y = (data >> 2) & 0x00001FFF
            polarity = (data >> 1) & 0x00000001

            if debug >= 3:
                print("ts->", timestamp)  # ok
                print("x-> ", x)
                print("y-> ", y)
                print("pol->", polarity)

            xaddr.append(x)
            yaddr.append(y)
            pol.append(polarity)
            timestamps.append(timestamp)
    if debug > 0:
        try:
            print("read {} event blocks".format(blocks_read))
            print ("read %i (~ %.2fM) AE events, duration= %.2fs" % (
                len(timestamps), len(timestamps) / float(10 ** 6), (timestamps[-1] - timestamps[0]) * td))
            n = 5
            print ("showing first %i:" % (n))
            print ("timestamps: %s \nX-addr: %s\nY-addr: %s\npolarity: %s" % (
                timestamps[0:n], xaddr[0:n], yaddr[0:n], pol[0:n]))
        except:
            print ("failed to print statistics")

    return timestamps, xaddr, yaddr, pol


def write_to_aedat31(filepath, fileheader, df):
    if df.ts.size == 0:
        return
    df.ts = df.ts - min(df.ts)
    # df = df[df.ts < 1e6] #TODO olol
    # df = df.iloc[:100000]

    header_len = 2 * 2 + 6 * 4  # bytes
    header_format = '<HHIIIIII'  # cp. struct.unpack and aedat header definition
    event_format = '<II'
    header_format = '<hhiiiiii'
    event_format = '<ii'
    event_type = 1
    event_source = 1
    event_size = 4 * 2  # bytes
    event_ts_offset = 4
    event_ts_overflow = 0
    max_block_size = 1000  # pow(2, 32) - 1
    df_rows = df.shape[0]
    c_s = 0
    with open(filepath, 'w+') as new_aedat:
        new_aedat.write(fileheader)
        for block in range(df_rows // max_block_size + 1):
            counter = 0
            block_size = min(max_block_size, df_rows - max_block_size * block)
            event_capacity = block_size
            event_number = block_size
            event_valid = block_size
            header = struct.pack(header_format, event_type, event_source, event_size, event_ts_offset,
                                 event_ts_overflow,
                                 event_capacity, event_number, event_valid)
            new_aedat.write(header)
            block_df = df[c_s:c_s + block_size]
            for e in block_df.itertuples():
                polarity_binary = '{:015b}{:015b}{:01b}{:01b}'.format(e.x, e.y, e.p, 1)
                ts_binary = '{:032b}'.format(e.ts)
                event = struct.pack(event_format, int(polarity_binary, 2), int(ts_binary, 2))
                new_aedat.write(event)
                counter += 1
            c_s += block_size
            assert counter == event_number, 'counter: {}, event_number: {}'.format(counter, event_number)
