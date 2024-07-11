//
//  MaskKernel.metal
//  YOLO
//
//  Created by 間嶋大輔 on 2024/07/03.
//  Copyright © 2024 Ultralytics. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

kernel void maskGenerationKernel(const device float* masks [[buffer(0)]],
                                 const device float* protos [[buffer(1)]],
                                 device uint8_t* maskPixels [[buffer(2)]],
                                 constant ushort &maskWidth [[buffer(3)]],
                                 constant ushort &maskHeight [[buffer(4)]],
                                 constant ushort &maskChannels [[buffer(5)]],
                                 constant float &threshold [[buffer(6)]],
                                 constant ushort4 &color [[buffer(7)]],
                                 ushort2 gid [[thread_position_in_grid]]) {
    if (gid.x >= maskWidth || gid.y >= maskHeight) {
        return;
    }

    uint index = gid.y * maskWidth + gid.x;
    float maskValue = masks[gid.y * maskChannels + gid.x]; // Adjust according to your data layout

    if (maskValue > threshold) {
        uint pixelIndex = index * 4;
        maskPixels[pixelIndex] = color.x;
        maskPixels[pixelIndex + 1] = color.y;
        maskPixels[pixelIndex + 2] = color.z;
        maskPixels[pixelIndex + 3] = color.w;
    }
}
