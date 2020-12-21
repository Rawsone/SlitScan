# ffmpeg must be installed. maybe by "apt install ffmpeg"
# custom_ffmpeg = "C:/Program Files/ffmpeg/bin" # that's for my ffmpeg installed in Windows
import cv2
from vidgear.gears import WriteGear
from collections import deque
import numpy as np
import argparse
from datetime import datetime
import numpy as np
from mpi4py import MPI
from os.path import splitext

def hms(timedelta):
    minutes, seconds = divmod(round(timedelta.total_seconds()), 60)
    hours, minutes = divmod(minutes, 60)
    return hours, minutes, seconds

def hms_str(timedelta):
    h, m, s = hms(timedelta)
    h_str = f"{h}h " if h else ""
    m_str = f"{m}m " if m or h else ""
    s_str = f"{s}s"
    return h_str + m_str + s_str

# move buf forward by n_frames through vid_gen video stream
# if stream ended, returns False, else True
def fwd(vid_gen, buf, n_frames):
    eos = False
    temp = []
    for frame in vid_gen:
        temp.append(frame)
        if len(temp) >= n_frames:
            break
    else:
        eos = True
    for i in range(n_frames):
        buf.popleft()
    buf.extend(temp) # if eos, len(buf) < buf_len possible
    return not eos


# draw one frame of output video
# and write the result into res.
# res is prepared like that: res = np.empty_like(video_frame)
def scan(buf, step, res, axis=1, rev=False):
    i = 0
    for frame in (reversed(buf) if rev else buf):
        if axis == 1:
            res[:, i:i+step] = frame[:, i:i+step]
        elif axis == 0:
            res[i:i+step, :] = frame[i:i+step, :]
        i += step

def videoreader_mod(cap, start=None, end=None, start_px=None, end_px=None, axis=0):
    """
    read video from "reader[start:end]"
    start, end are indices of frames (starting from 0), end is excluded
    """
    i = start or 0
    if start is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    while end is None or i < end:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame[start_px:end_px, :] if axis == 0 else frame[:, start_px:end_px]
        i += 1

def parse_command_line():
    parser = argparse.ArgumentParser(description='make slitscan frame sequence from video')
    parser.add_argument('filename', type=str,
                        help='input video path')

    parser.add_argument('out_name', type=str,
                        help='output video path')

    parser.add_argument('-s', '--step', type=int, default=1, dest='step',
                        help='slice length (step)')

    parser.add_argument('--timediv', type=int, dest='grid_w',
                        help='divide video length by TIMEDIV processes')

    parser.add_argument('--imgdiv', type=int, dest='grid_h',
                        help='divide frame width or height (depending on --dir) by IMGDIV processes')

    parser.add_argument('-d', '--dir', type=str,
                        choices=['right', 'left', 'up', 'down'], default='right',
                        dest='dir',
                        help="scanning dir (default=right)")

    parser.add_argument('-r', '--outrate', type=int, default=1, dest='outrate', help=
"""decrease framerate of output video OUTRATE times in following way:
if we don't decrease framerate of output video, it would consist of n frames V[0],V[1],...,V[n-1]
(n = m - buf_len + 1, m = number of frames in input video fragment)
let OUTRATE=r.
if we decrease framerate of output video, it would consist of frames V[0],V[r],V[2r],... (while i*r<=n-1)
""")

    parser.add_argument('-c', '--crf', type=int, default=None, dest='crf', help=
"""libx264 CRF, defines quality of compressed mp4 output video.
usually with crf=18 quality loss isn't visible. default is 23.""")

    return parser.parse_args()

def ceildiv(a, b):
    return -(-a//b)

import glob
import os
import subprocess
from datetime import datetime

def concat(file_names, out_name):
    elenco_video = file_names
    elenco_file_temp = []
    for f in elenco_video:
        file = "temp" + str(elenco_video.index(f) + 1) + ".ts"
#         os.system("ffmpeg -i " + f + " -c copy -bsf:v h264_mp4toannexb -f mpegts " + file)
        cmd = f'ffmpeg -i "{f}" -c copy -bsf:v h264_mp4toannexb -f mpegts "{file}"'
        print(f"[concat -> {out_name}] launching command 1: {cmd}")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        output, err = p.communicate()
        p_status = p.wait()
        elenco_file_temp.append(file)
    s = '|'.join(elenco_file_temp)
    stringa = f'ffmpeg -i "concat:{s}" -c copy -bsf:a aac_adtstoasc "{out_name}"'
    print(f"[concat -> {out_name}] launching command 2: {stringa}")
    p = subprocess.Popen(stringa, stdout=subprocess.PIPE, shell=True)
    output, err = p.communicate()
    p_status = p.wait()
    for f in elenco_file_temp:
        if os.path.isfile(f):
            os.remove(f)
        else:
            print(f"cant remove file {f}, not found")

def stack(file_names, out_name, axis):
    #make command
    command = "ffmpeg"
    for f_name in file_names:
        command += " -i " + '"' + f_name + '"'
    stack_type = "vstack" if axis == 0 else "hstack"
    command += f" -filter_complex " +  stack_type + f"={len(file_names)} " + '"' + out_name + '"'
    print(f"[stack -> {out_name}] launching command: {command}")

    #submit it
    p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    #some checks here?
    output, err = p.communicate()
    p_status = p.wait()

def slitscan(filename, out_name, dir, step=1, outrate=1, start=0, end=-1, startpx=0, endpx=-1, outlen=-1, crf=18, print_prefix=""):
    success = True
    exc = None
    axis, rev = int(dir in ('right', 'left')), dir in ('left', 'down')
    if start == 0:
        start = None
    if end == -1:
        end = None
    assert(end is None or start is None or end > start)
    t1 = datetime.now()
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        raise ValueError(f"couldn't open file: {filename}")
    if outlen == -1:
        outlen = None
    try:
        vid = videoreader_mod(cap, start, end, startpx, endpx)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(print_prefix, "w, h = ", width, height)
        if axis == 1:
            side_len = width
        elif axis == 0:
            side_len = height
        if startpx == 0:
            startpx = None
        if endpx == -1:
            endpx = None
        if startpx is not None:
            if endpx is None:
                side_len = side_len - startpx
                print(print_prefix, "startpx, no endpx: side_len =", side_len)
            else:
                side_len = endpx - startpx
                print(print_prefix, "startpx, endpx: side_len =", side_len)
        elif endpx is not None:
            side_len = endpx
            print(print_prefix, "no startpx, endpx: side_len =", side_len)
        n_frames_full = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # how many frames in input video file
        n_frames = (end or n_frames_full) - (start or 0) # how many frames will be read
        buf_len = -(-side_len // step) # = ceil(side_len / step)
        n_out_frames_full = n_frames - buf_len + 1 # that would be length of output video if outrate==1
        n_out_frames = (n_out_frames_full - 1) // outrate + 1 # real length of output video
        if outlen is not None:
            n_out_frames = outlen
        frame = None
        for frame in vid: # get first frame to know its dtype and shape
            break
        if frame is None:
            print(print_prefix, "video reader is empty!")
            return False, exc, 0, n_out_frames
        buf = deque([frame]) # FIX: forgot [frame] here in previous versions of code!
                             # seems like the first frame of video was never used then
        print(print_prefix, f"number of input frames: {n_frames}")
        print(print_prefix, f"number of output frames: {n_out_frames}")
        print(print_prefix, f"buflen: {buf_len}")
        print(print_prefix, f"side_len: {side_len}")
        print(print_prefix, f"outlen: {outlen or -1}")
        if n_frames < buf_len:
            raise ValueError("window size is larger than video length. try bigger step or longer video")
        output_params = {"-vcodec" : "libx264"}
        if crf:
            output_params["-crf"] = str(crf)
        writer = WriteGear(output_filename=out_name, logging=False, **output_params)
        try:
            for frame in vid:
                buf.append(frame)
                print(print_prefix, f"init window: {len(buf)} / {buf_len}" + " "*7, end="\r")
                if len(buf) >= buf_len:
                    break
            print()
            n_written = 0
            eos = False
            res = np.zeros_like(frame)
            # temp = np.zeros((outrate, *frame.shape), dtype=frame.dtype)
            print(print_prefix, "shape:", frame.shape)
            while True:
                res[:] = 0 # just debug
                scan(buf, step, res, axis=axis, rev=rev)
                writer.write(res)
                # # check for 'q' key if pressed
                # key = cv2.waitKey(1) & 0xFF
                # if key == ord("q"):
                #     break
                n_written += 1
                s1 = "frame=xxxxx " # fragment of ffmpeg output to be overwritten by our print
                s2 = f"{n_written}/{n_out_frames}" # to write over ffmpeg output
                print(print_prefix, f"{s2.ljust(len(s1))}", end="\r")
                # print(f"{s2.ljust(len(s1))}")
                if eos or (outlen is not None and n_written >= outlen):
                    # we can't read any more frames from video reader
                    break
                # temp[:] = 0 # just debug
                eos = not fwd(vid, buf, outrate)
                if eos and len(buf) != buf_len:
                    # we can't write another frame because we need full buf for it,
                    # so our output video is ended
                    break
        except Exception as e:
            success = False
            exc = e
            print(print_prefix, "\nEXCEPTION:\n", e)
            print(print_prefix, "video is written but it may be incomplete")
        finally:
            print()
            writer.close()  # safely close writer
            del buf
            done_time_str = hms_str(datetime.now() - t1)
            print(print_prefix, f"done in {done_time_str}")
            print(print_prefix, f"{n_written}/{n_out_frames} frames written/expected")
    except Exception as e:
        success = False
        exc = e
        print(print_prefix, "\nEXCEPTION:\n", e)
    finally:
        cv2.destroyAllWindows() # close output window
        cap.release() # safely close video stream
    return success, exc, n_written, n_out_frames

t1 = datetime.now()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#parse arguments
args = parse_command_line()
filename = args.filename
out_name_full = args.out_name
out_name, ext = splitext(out_name_full)
step = args.step
dir = args.dir
axis = int(dir in ('right', 'left'))
r = args.outrate
crf = args.crf
grid_w = args.grid_w
grid_h = args.grid_h
nproc_time = grid_w # synonym
nproc_img = grid_h # synonym

assert(size == grid_w * grid_h)

#creating grid
grid2d = comm.Create_cart(dims=[grid_h, grid_w], periods=[False, False], reorder=False)
#creating column communicator
row, col = grid2d.coords
group_ranks = [col + i * grid_w for i in range(grid_h)]
column_gr = comm.group.Incl(group_ranks)
column_comm = comm.Create_group(column_gr)

#rank == 0 => root of the grid
#row == 0 => root of the column

cap = cv2.VideoCapture(filename)
if not cap.isOpened():
    raise ValueError(f"couldn't open file: {filename}")
try:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    m = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # num of input frames
    if axis == 1:
        side = width
    elif axis == 0:
        side = height
except Exception as e:
    print("\nEXCEPTION:\n", e)
finally:
    cap.release()

b = ceildiv(side, step) # buffer length
n = (m-b)//r + 1 # number of output frames
m_used = ceildiv(m - b, r)*r + b
side_part = ceildiv(side, nproc_img)
time_part = ceildiv(m_used, nproc_time)
if rank == 0:
    print("px_divisions:", [side_part*i for i in range(nproc_img)] + [side])
if rank == 0:
    print("time_divisions:", [time_part*i for i in range(nproc_time)] + [m_used])
f_start = ceildiv(col*time_part,r)*r
f_stop = min(m_used, (col+1)*time_part + b - 1)
print_prefix = f"[{row}, {col}]"
n_timepart = (f_stop-f_start-b)//r+1
b_part = ceildiv(side_part, step)
start_shift = row*b_part
startpx = row*side_part
endpx = min((row+1)*side_part, side)
print(print_prefix, f"start={f_start:5}, stop={f_stop:5}, time_part={time_part:5}, b={b:5}, n_timepart={n_timepart:5} ; start_shift={start_shift:5}, startpx={startpx:5}, endpx={endpx:5}, b_part={b_part:5}")

#1 main calculations
res_name = out_name + f'_{row}' + f'_{col}' + ext
success, e, n_written, n_expected = slitscan(filename, res_name, dir, step=step, outrate=r, start=f_start+start_shift, end=-1,
         startpx=startpx, endpx=endpx, outlen=n_timepart, print_prefix=print_prefix)

print(print_prefix, f"executed slitscan into {res_name}, succ={success}, n_wr={n_written}, n_exp={n_expected}")
if e is not None:
    print(print_prefix, f"and also got exception:\n{e}  <end of exception>")

#2 column root collects video part
column_comm.Barrier()
if row == 0:
    #reads these files
    read_names = [out_name + f'_{i}' + f'_{col}' + ext for i in range(grid_h)]
    #creates this file
    res_name = out_name + f'_{col}' + ext
    #execute gather function
    stack(read_names,res_name, axis)
    #delete these files
    del_names = read_names
    for f in del_names:
        if os.path.isfile(f):
            os.remove(f)
        else:
            print(f"cant remove file {f}, not found")
    print(print_prefix, f"[col {col} root] my file = {res_name}, del = {del_names}")


#3 grid root collects everything
comm.Barrier()
if rank == 0:
    #reads these files
    read_names = [out_name + f'_{i}' + ext for i in range(grid_w)]
    #creates this file
    res_name = out_name + ext
    #execute final gather function
    concat(read_names, res_name)
    #deletes these files
    del_names = read_names
    for f in del_names:
        if os.path.isfile(f):
            os.remove(f)
        else:
            print(f"cant remove file {f}, not found")
    print(print_prefix, f"[world root] my file = {res_name}, del = {del_names}")
    t2 = datetime.now()
    print(t2-t1)
    print(f'time={(t2-t1).total_seconds()}')
    print(f'buf_per_proc={b_part}')
    print(f'buf_per_total={b_part*size}')
