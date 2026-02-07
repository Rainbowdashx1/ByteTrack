# Bytetrack

## Description

**ByteTrack** is a multi-object tracking approach used to assign persistent IDs to objects detected by an AI model (e.g., YOLO) across frames. In practice, it takes the detections per frame (bounding boxes + confidence) and links them over time, so you can say “this is the same person/car as in the previous frames” and keep a stable track ID while it moves, gets partially occluded, or briefly disappears.

![TestBytetrack](https://raw.githubusercontent.com/Rainbowdashx1/Bytetrack/master/Bytetrack.gif)

## Current Status (“Dirty” ByteTrack)

At the moment, this is a **“dirty” ByteTrack**: it’s not optimized and is implemented in a straightforward way just to work for now. It serves as an initial functional baseline, with the intention to refactor and optimize it later (performance, memory usage, structure, and code cleanliness).

