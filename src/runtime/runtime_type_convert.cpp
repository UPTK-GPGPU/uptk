#include "runtime.hpp"

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

cudaAccessProperty UPTKAccessPropertyTocudaAccessProperty(enum UPTKAccessProperty para) {
    switch (para) {
        case UPTKAccessPropertyNormal:
            return cudaAccessPropertyNormal;
        case UPTKAccessPropertyPersisting:
            return cudaAccessPropertyPersisting;
        case UPTKAccessPropertyStreaming:
            return cudaAccessPropertyStreaming;
        default:
              ERROR_INVALID_ENUM();
    }
}

enum UPTKAccessProperty cudaAccessPropertyToUPTKAccessProperty(cudaAccessProperty para) {
    switch (para) {
        case cudaAccessPropertyNormal:
            return UPTKAccessPropertyNormal;
        case cudaAccessPropertyPersisting:
            return UPTKAccessPropertyPersisting;
        case cudaAccessPropertyStreaming:
            return UPTKAccessPropertyStreaming;
        default:
              ERROR_INVALID_ENUM();
    }
}

enum cudaTextureAddressMode UPTKTextureAddressModeTocudaTextureAddressMode(enum UPTKTextureAddressMode para) {
    switch (para) {
        case UPTKAddressModeBorder:
            return cudaAddressModeBorder;
        case UPTKAddressModeClamp:
            return cudaAddressModeClamp;
        case UPTKAddressModeMirror:
            return cudaAddressModeMirror;
        case UPTKAddressModeWrap:
            return cudaAddressModeWrap;
        default:
            ERROR_INVALID_ENUM();
    }
}

enum cudaSurfaceBoundaryMode UPTKSurfaceBoundaryModeTocudaSurfaceBoundaryMode(enum UPTKSurfaceBoundaryMode para) {
    switch (para) {
        case UPTKBoundaryModeClamp:
            return cudaBoundaryModeClamp;
        case UPTKBoundaryModeTrap:
            return cudaBoundaryModeTrap;
        case UPTKBoundaryModeZero:
            return cudaBoundaryModeZero;
        default:
            ERROR_INVALID_ENUM();
    }
}

cudaChannelFormatKind UPTKChannelFormatKindTocudaChannelFormatKind(enum UPTKChannelFormatKind para) {
    switch (para) {
        case UPTKChannelFormatKindFloat:
            return cudaChannelFormatKindFloat;
        case UPTKChannelFormatKindNone:
            return cudaChannelFormatKindNone;
        case UPTKChannelFormatKindSigned:
            return cudaChannelFormatKindSigned;
        case UPTKChannelFormatKindUnsigned:
            return cudaChannelFormatKindUnsigned;
        default:
            ERROR_INVALID_ENUM();
    }
}

enum UPTKChannelFormatKind cudaChannelFormatKindToUPTKChannelFormatKind(cudaChannelFormatKind para) {
    switch (para) {
        case cudaChannelFormatKindFloat:
            return UPTKChannelFormatKindFloat;
        case cudaChannelFormatKindNone:
            return UPTKChannelFormatKindNone;
        case cudaChannelFormatKindSigned:
            return UPTKChannelFormatKindSigned;
        case cudaChannelFormatKindUnsigned:
            return UPTKChannelFormatKindUnsigned;
        default:
            ERROR_INVALID_ENUM();
    }
}

enum cudaComputeMode UPTKComputeModeTocudaComputeMode(enum UPTKComputeMode para) {
    switch (para) {
        case UPTKComputeModeDefault:
            return cudaComputeModeDefault;
        case UPTKComputeModeExclusive:
            return cudaComputeModeExclusive;
        case UPTKComputeModeExclusiveProcess:
            return cudaComputeModeExclusiveProcess;
        case UPTKComputeModeProhibited:
            return cudaComputeModeProhibited;
        default:
            ERROR_INVALID_ENUM();
    }
}

cudaDeviceAttr UPTKDeviceAttrTocudaDeviceAttr(enum UPTKDeviceAttr para) {
    switch (para) {
        case UPTKDevAttrAsyncEngineCount:
            return cudaDevAttrAsyncEngineCount;
        case UPTKDevAttrCanMapHostMemory:
            return cudaDevAttrCanMapHostMemory;
        case UPTKDevAttrCanUseHostPointerForRegisteredMem:
            return cudaDevAttrCanUseHostPointerForRegisteredMem;
        case UPTKDevAttrClockRate:
            return cudaDevAttrClockRate;
        case UPTKDevAttrComputeCapabilityMajor:
            return cudaDevAttrComputeCapabilityMajor;
        case UPTKDevAttrComputeCapabilityMinor:
            return cudaDevAttrComputeCapabilityMinor;
        case UPTKDevAttrComputeMode:
            return cudaDevAttrComputeMode;
        case UPTKDevAttrComputePreemptionSupported:
            return cudaDevAttrComputePreemptionSupported;
        case UPTKDevAttrConcurrentKernels:
            return cudaDevAttrConcurrentKernels;
        case UPTKDevAttrConcurrentManagedAccess:
            return cudaDevAttrConcurrentManagedAccess;
        case UPTKDevAttrCooperativeLaunch:
            return cudaDevAttrCooperativeLaunch;
        case UPTKDevAttrCooperativeMultiDeviceLaunch:
            return cudaDevAttrCooperativeMultiDeviceLaunch;
        case UPTKDevAttrDirectManagedMemAccessFromHost:
            return cudaDevAttrDirectManagedMemAccessFromHost;
        case UPTKDevAttrEccEnabled:
            return cudaDevAttrEccEnabled;
        case UPTKDevAttrGlobalL1CacheSupported:
            return cudaDevAttrGlobalL1CacheSupported;
        case UPTKDevAttrGlobalMemoryBusWidth:
            return cudaDevAttrGlobalMemoryBusWidth;
        case UPTKDevAttrGpuOverlap:
            return cudaDevAttrGpuOverlap;
        case UPTKDevAttrHostNativeAtomicSupported:
            return cudaDevAttrHostNativeAtomicSupported;
        case UPTKDevAttrHostRegisterSupported:
            return cudaDevAttrHostRegisterSupported;
        case UPTKDevAttrIntegrated:
            return cudaDevAttrIntegrated;
        case UPTKDevAttrIsMultiGpuBoard:
            return cudaDevAttrIsMultiGpuBoard;
        case UPTKDevAttrKernelExecTimeout:
            return cudaDevAttrKernelExecTimeout;
        case UPTKDevAttrL2CacheSize:
            return cudaDevAttrL2CacheSize;
        case UPTKDevAttrLocalL1CacheSupported:
            return cudaDevAttrLocalL1CacheSupported;
        case UPTKDevAttrManagedMemory:
            return cudaDevAttrManagedMemory;
        case UPTKDevAttrMaxBlockDimX:
            return cudaDevAttrMaxBlockDimX;
        case UPTKDevAttrMaxBlockDimY:
            return cudaDevAttrMaxBlockDimY;
        case UPTKDevAttrMaxBlockDimZ:
            return cudaDevAttrMaxBlockDimZ;
        case UPTKDevAttrMaxGridDimX:
            return cudaDevAttrMaxGridDimX;
        case UPTKDevAttrMaxGridDimY:
            return cudaDevAttrMaxGridDimY;
        case UPTKDevAttrMaxGridDimZ:
            return cudaDevAttrMaxGridDimZ;
        case UPTKDevAttrMaxPitch:
            return cudaDevAttrMaxPitch;
        case UPTKDevAttrMaxRegistersPerBlock:
            return cudaDevAttrMaxRegistersPerBlock;
        case UPTKDevAttrMaxRegistersPerMultiprocessor:
            return cudaDevAttrMaxRegistersPerMultiprocessor;
        case UPTKDevAttrMaxSharedMemoryPerBlock:
            return cudaDevAttrMaxSharedMemoryPerBlock;
        // cudaDeviceAttributeSharedMemPerBlockOptin actually returns 0.
        // Replace with cudaDeviceAttributeMaxSharedMemoryPerBlock.
        case UPTKDevAttrMaxSharedMemoryPerBlockOptin:
            return cudaDevAttrMaxSharedMemoryPerBlockOptin;
        case UPTKDevAttrMaxSharedMemoryPerMultiprocessor:
            return cudaDevAttrMaxSharedMemoryPerMultiprocessor;
//         case UPTKDevAttrMaxSurface1DLayeredLayers:
//             return cudaDeviceAttributeMaxSurface1DLayeredLayers;
        // case UPTKDevAttrMaxSurface1DLayeredWidth:
        //     return cudaDeviceAttributeMaxSurface1DLayered;
        case UPTKDevAttrMaxSurface1DWidth:
            return cudaDevAttrMaxSurface1DWidth;
        // case UPTKDevAttrMaxSurface2DHeight:
        //     return cudaDeviceAttributeMaxSurface2D;
        // case UPTKDevAttrMaxSurface2DLayeredHeight:
        //     return cudaDeviceAttributeMaxSurface2DLayered;
//         case UPTKDevAttrMaxSurface2DLayeredLayers:
//             return cudaDeviceAttributeMaxSurface2DLayeredLayers;
        // case UPTKDevAttrMaxSurface2DLayeredWidth:
        //     return cudaDeviceAttributeMaxSurface2DLayered;
        // case UPTKDevAttrMaxSurface2DWidth:
        //     return cudaDeviceAttributeMaxSurface2D;
        // case UPTKDevAttrMaxSurface3DDepth:
        //     return cudaDeviceAttributeMaxSurface3D;
        // case UPTKDevAttrMaxSurface3DHeight:
        //     return cudaDeviceAttributeMaxSurface3D;
        // case UPTKDevAttrMaxSurface3DWidth:
        //     return cudaDeviceAttributeMaxSurface3D;
//         case UPTKDevAttrMaxSurfaceCubemapLayeredLayers:
//             return cudaDeviceAttributeMaxSurfaceCubemapLayeredLayers;
        // case UPTKDevAttrMaxSurfaceCubemapLayeredWidth:
        //     return cudaDeviceAttributeMaxSurfaceCubemapLayered;
        // case UPTKDevAttrMaxSurfaceCubemapWidth:
        //     return cudaDeviceAttributeMaxSurfaceCubemap;
//         case UPTKDevAttrMaxTexture1DLayeredLayers:
//             return cudaDeviceAttributeMaxTexture1DLayeredLayers;
        // case UPTKDevAttrMaxTexture1DLayeredWidth:
        //     return cudaDeviceAttributeMaxTexture1DLayered;
        case UPTKDevAttrMaxTexture1DLinearWidth:
            return cudaDevAttrMaxTexture1DLinearWidth;
        case UPTKDevAttrMaxTexture1DMipmappedWidth:
            return cudaDevAttrMaxTexture1DMipmappedWidth;
        case UPTKDevAttrMaxTexture1DWidth:
            return cudaDevAttrMaxTexture1DWidth;
        // case UPTKDevAttrMaxTexture2DGatherHeight:
        //     return cudaDeviceAttributeMaxTexture2DGather;
        // case UPTKDevAttrMaxTexture2DGatherWidth:
        //     return cudaDeviceAttributeMaxTexture2DGather;
        case UPTKDevAttrMaxTexture2DHeight:
            return cudaDevAttrMaxTexture2DHeight;
        // case UPTKDevAttrMaxTexture2DLayeredHeight:
        //     return cudaDeviceAttributeMaxTexture2DLayered;
//         case UPTKDevAttrMaxTexture2DLayeredLayers:
//             return cudaDeviceAttributeMaxTexture2DLayeredLayers;
        // case UPTKDevAttrMaxTexture2DLayeredWidth:
        //     return cudaDeviceAttributeMaxTexture2DLayered;
        // case UPTKDevAttrMaxTexture2DLinearHeight:
        //     return cudaDeviceAttributeMaxTexture2DLinear;
        // case UPTKDevAttrMaxTexture2DLinearPitch:
        //     return cudaDeviceAttributeMaxTexture2DLinear;
        // case UPTKDevAttrMaxTexture2DLinearWidth:
        //     return cudaDeviceAttributeMaxTexture2DLinear;
        // case UPTKDevAttrMaxTexture2DMipmappedHeight:
        //     return cudaDeviceAttributeMaxTexture2DMipmap;
        // case UPTKDevAttrMaxTexture2DMipmappedWidth:
        //     return cudaDeviceAttributeMaxTexture2DMipmap;
        case UPTKDevAttrMaxTexture2DWidth:
            return cudaDevAttrMaxTexture2DWidth;
        case UPTKDevAttrMaxTexture3DDepth:
            return cudaDevAttrMaxTexture3DDepth;
        // case UPTKDevAttrMaxTexture3DDepthAlt:
        //     return cudaDeviceAttributeMaxTexture3DAlt;
        case UPTKDevAttrMaxTexture3DHeight:
            return cudaDevAttrMaxTexture3DHeight;
        // case UPTKDevAttrMaxTexture3DHeightAlt:
        //     return cudaDeviceAttributeMaxTexture3DAlt;
        case UPTKDevAttrMaxTexture3DWidth:
            return cudaDevAttrMaxTexture3DWidth;
        // case UPTKDevAttrMaxTexture3DWidthAlt:
        //     return cudaDeviceAttributeMaxTexture3DAlt;
//         case UPTKDevAttrMaxTextureCubemapLayeredLayers:
//             return cudaDeviceAttributeMaxTextureCubemapLayeredLayers;
        // case UPTKDevAttrMaxTextureCubemapLayeredWidth:
        //     return cudaDeviceAttributeMaxTextureCubemapLayered;
        case UPTKDevAttrMaxTextureCubemapWidth:
            return cudaDevAttrMaxTextureCubemapWidth;
        case UPTKDevAttrMaxThreadsPerBlock:
            return cudaDevAttrMaxThreadsPerBlock;
        case UPTKDevAttrMaxThreadsPerMultiProcessor:
            return cudaDevAttrMaxThreadsPerMultiProcessor;
        case UPTKDevAttrMemoryClockRate:
            return cudaDevAttrMemoryClockRate;
        case UPTKDevAttrMultiGpuBoardGroupID:
            return cudaDevAttrMultiGpuBoardGroupID;
        case UPTKDevAttrMultiProcessorCount:
            return cudaDevAttrMultiProcessorCount;
        case UPTKDevAttrPageableMemoryAccess:
            return cudaDevAttrPageableMemoryAccess;
        case UPTKDevAttrPageableMemoryAccessUsesHostPageTables:
            return cudaDevAttrPageableMemoryAccessUsesHostPageTables;
        case UPTKDevAttrPciBusId:
            return cudaDevAttrPciBusId;
        case UPTKDevAttrPciDeviceId:
            return cudaDevAttrPciDeviceId;
        case UPTKDevAttrPciDomainId:
            return cudaDevAttrPciDomainId;
//         case UPTKDevAttrReserved92:
//             return cudaDeviceAttributeCanUseStreamMemOps;
//         case UPTKDevAttrReserved93:
//             return cudaDeviceAttributeCanUse64BitStreamMemOps;
        case UPTKDevAttrReserved94:
            return cudaDevAttrReserved94;
        // case UPTKDevAttrSingleToDoublePrecisionPerfRatio:
        //     return cudaDeviceAttributeSingleToDoublePrecisionPerfRatio;
        case UPTKDevAttrStreamPrioritiesSupported:
            return cudaDevAttrStreamPrioritiesSupported;
        case UPTKDevAttrSurfaceAlignment:
            return cudaDevAttrSurfaceAlignment;
        // case UPTKDevAttrTccDriver:
        //     return cudaDeviceAttributeTccDriver;
        case UPTKDevAttrTextureAlignment:
            return cudaDevAttrTextureAlignment;
        case UPTKDevAttrTexturePitchAlignment:
            return cudaDevAttrTexturePitchAlignment;
        case UPTKDevAttrTotalConstantMemory:
            return cudaDevAttrTotalConstantMemory;
        case UPTKDevAttrUnifiedAddressing:
            return cudaDevAttrUnifiedAddressing;
        case UPTKDevAttrWarpSize:
            return cudaDevAttrWarpSize;
        case UPTKDevAttrMemoryPoolsSupported:
            return cudaDevAttrMemoryPoolsSupported;
        case UPTKDevAttrReservedSharedMemoryPerBlock:
            return cudaDevAttrReservedSharedMemoryPerBlock;
        case UPTKDevAttrMaxBlocksPerMultiprocessor:
            return cudaDevAttrMaxBlocksPerMultiprocessor;
        default:
            ERROR_INVALID_OR_UNSUPPORTED_ENUM();
    }
}

cudaDeviceP2PAttr UPTKDeviceP2PAttrTocudaDeviceP2PAttr(enum UPTKDeviceP2PAttr para) {
    switch (para) {
        case UPTKDevP2PAttrAccessSupported:
            return cudaDevP2PAttrAccessSupported;
        case UPTKDevP2PAttrNativeAtomicSupported:
            return cudaDevP2PAttrNativeAtomicSupported;
        case UPTKDevP2PAttrPerformanceRank:
            return cudaDevP2PAttrPerformanceRank;
        case UPTKDevP2PAttrCudaArrayAccessSupported:
            return cudaDevP2PAttrCudaArrayAccessSupported;
        default:
            ERROR_INVALID_ENUM();
    }
}

// enum cudaEglColorFormat UPTKEglColorFormatTocudaEglColorFormat(UPTKEglColorFormat para) {
//     switch (para) {
//         case UPTKEglColorFormatA:
//             return cudaEglColorFormatA;
//         case UPTKEglColorFormatABGR:
//             return cudaEglColorFormatABGR;
//         case UPTKEglColorFormatARGB:
//             return cudaEglColorFormatARGB;
//         case UPTKEglColorFormatAYUV:
//             return cudaEglColorFormatAYUV;
//         case UPTKEglColorFormatAYUV_ER:
//             return cudaEglColorFormatAYUV_ER;
//         case UPTKEglColorFormatBGR:
//             return cudaEglColorFormatBGR;
//         case UPTKEglColorFormatBGRA:
//             return cudaEglColorFormatBGRA;
//         case UPTKEglColorFormatBayer10BGGR:
//             return cudaEglColorFormatBayer10BGGR;
//         case UPTKEglColorFormatBayer10GBRG:
//             return cudaEglColorFormatBayer10GBRG;
//         case UPTKEglColorFormatBayer10GRBG:
//             return cudaEglColorFormatBayer10GRBG;
//         case UPTKEglColorFormatBayer10RGGB:
//             return cudaEglColorFormatBayer10RGGB;
//         case UPTKEglColorFormatBayer12BGGR:
//             return cudaEglColorFormatBayer12BGGR;
//         case UPTKEglColorFormatBayer12GBRG:
//             return cudaEglColorFormatBayer12GBRG;
//         case UPTKEglColorFormatBayer12GRBG:
//             return cudaEglColorFormatBayer12GRBG;
//         case UPTKEglColorFormatBayer12RGGB:
//             return cudaEglColorFormatBayer12RGGB;
//         case UPTKEglColorFormatBayer14BGGR:
//             return cudaEglColorFormatBayer14BGGR;
//         case UPTKEglColorFormatBayer14GBRG:
//             return cudaEglColorFormatBayer14GBRG;
//         case UPTKEglColorFormatBayer14GRBG:
//             return cudaEglColorFormatBayer14GRBG;
//         case UPTKEglColorFormatBayer14RGGB:
//             return cudaEglColorFormatBayer14RGGB;
//         case UPTKEglColorFormatBayer20BGGR:
//             return cudaEglColorFormatBayer20BGGR;
//         case UPTKEglColorFormatBayer20GBRG:
//             return cudaEglColorFormatBayer20GBRG;
//         case UPTKEglColorFormatBayer20GRBG:
//             return cudaEglColorFormatBayer20GRBG;
//         case UPTKEglColorFormatBayer20RGGB:
//             return cudaEglColorFormatBayer20RGGB;
//         case UPTKEglColorFormatBayerBGGR:
//             return cudaEglColorFormatBayerBGGR;
//         case UPTKEglColorFormatBayerGBRG:
//             return cudaEglColorFormatBayerGBRG;
//         case UPTKEglColorFormatBayerGRBG:
//             return cudaEglColorFormatBayerGRBG;
//         case UPTKEglColorFormatBayerIspBGGR:
//             return cudaEglColorFormatBayerIspBGGR;
//         case UPTKEglColorFormatBayerIspGBRG:
//             return cudaEglColorFormatBayerIspGBRG;
//         case UPTKEglColorFormatBayerIspGRBG:
//             return cudaEglColorFormatBayerIspGRBG;
//         case UPTKEglColorFormatBayerIspRGGB:
//             return cudaEglColorFormatBayerIspRGGB;
//         case UPTKEglColorFormatBayerRGGB:
//             return cudaEglColorFormatBayerRGGB;
//         case UPTKEglColorFormatL:
//             return cudaEglColorFormatL;
//         case UPTKEglColorFormatR:
//             return cudaEglColorFormatR;
//         case UPTKEglColorFormatRG:
//             return cudaEglColorFormatRG;
//         case UPTKEglColorFormatRGB:
//             return cudaEglColorFormatRGB;
//         case UPTKEglColorFormatRGBA:
//             return cudaEglColorFormatRGBA;
//         case UPTKEglColorFormatUYVY422:
//             return cudaEglColorFormatUYVY422;
//         case UPTKEglColorFormatUYVY_ER:
//             return cudaEglColorFormatUYVY_ER;
//         case UPTKEglColorFormatVYUY_ER:
//             return cudaEglColorFormatVYUY_ER;
//         case UPTKEglColorFormatY10V10U10_420SemiPlanar:
//             return cudaEglColorFormatY10V10U10_420SemiPlanar;
//         case UPTKEglColorFormatY10V10U10_444SemiPlanar:
//             return cudaEglColorFormatY10V10U10_444SemiPlanar;
//         case UPTKEglColorFormatY12V12U12_420SemiPlanar:
//             return cudaEglColorFormatY12V12U12_420SemiPlanar;
//         case UPTKEglColorFormatY12V12U12_444SemiPlanar:
//             return cudaEglColorFormatY12V12U12_444SemiPlanar;
//         case UPTKEglColorFormatYUV420Planar:
//             return cudaEglColorFormatYUV420Planar;
//         case UPTKEglColorFormatYUV420Planar_ER:
//             return cudaEglColorFormatYUV420Planar_ER;
//         case UPTKEglColorFormatYUV420SemiPlanar:
//             return cudaEglColorFormatYUV420SemiPlanar;
//         case UPTKEglColorFormatYUV420SemiPlanar_ER:
//             return cudaEglColorFormatYUV420SemiPlanar_ER;
//         case UPTKEglColorFormatYUV422Planar:
//             return cudaEglColorFormatYUV422Planar;
//         case UPTKEglColorFormatYUV422Planar_ER:
//             return cudaEglColorFormatYUV422Planar_ER;
//         case UPTKEglColorFormatYUV422SemiPlanar:
//             return cudaEglColorFormatYUV422SemiPlanar;
//         case UPTKEglColorFormatYUV422SemiPlanar_ER:
//             return cudaEglColorFormatYUV422SemiPlanar_ER;
//         case UPTKEglColorFormatYUV444Planar:
//             return cudaEglColorFormatYUV444Planar;
//         case UPTKEglColorFormatYUV444Planar_ER:
//             return cudaEglColorFormatYUV444Planar_ER;
//         case UPTKEglColorFormatYUV444SemiPlanar:
//             return cudaEglColorFormatYUV444SemiPlanar;
//         case UPTKEglColorFormatYUV444SemiPlanar_ER:
//             return cudaEglColorFormatYUV444SemiPlanar_ER;
//         case UPTKEglColorFormatYUVA_ER:
//             return cudaEglColorFormatYUVA_ER;
//         case UPTKEglColorFormatYUV_ER:
//             return cudaEglColorFormatYUV_ER;
//         case UPTKEglColorFormatYUYV422:
//             return cudaEglColorFormatYUYV422;
//         case UPTKEglColorFormatYUYV_ER:
//             return cudaEglColorFormatYUYV_ER;
//         case UPTKEglColorFormatYVU420Planar:
//             return cudaEglColorFormatYVU420Planar;
//         case UPTKEglColorFormatYVU420Planar_ER:
//             return cudaEglColorFormatYVU420Planar_ER;
//         case UPTKEglColorFormatYVU420SemiPlanar:
//             return cudaEglColorFormatYVU420SemiPlanar;
//         case UPTKEglColorFormatYVU420SemiPlanar_ER:
//             return cudaEglColorFormatYVU420SemiPlanar_ER;
//         case UPTKEglColorFormatYVU422Planar:
//             return cudaEglColorFormatYVU422Planar;
//         case UPTKEglColorFormatYVU422Planar_ER:
//             return cudaEglColorFormatYVU422Planar_ER;
//         case UPTKEglColorFormatYVU422SemiPlanar:
//             return cudaEglColorFormatYVU422SemiPlanar;
//         case UPTKEglColorFormatYVU422SemiPlanar_ER:
//             return cudaEglColorFormatYVU422SemiPlanar_ER;
//         case UPTKEglColorFormatYVU444Planar:
//             return cudaEglColorFormatYVU444Planar;
//         case UPTKEglColorFormatYVU444Planar_ER:
//             return cudaEglColorFormatYVU444Planar_ER;
//         case UPTKEglColorFormatYVU444SemiPlanar:
//             return cudaEglColorFormatYVU444SemiPlanar;
//         case UPTKEglColorFormatYVU444SemiPlanar_ER:
//             return cudaEglColorFormatYVU444SemiPlanar_ER;
//         case UPTKEglColorFormatYVYU_ER:
//             return cudaEglColorFormatYVYU_ER;
//         default:
//             ERROR_INVALID_ENUM();
//     }
// }

// enum cudaEglFrameType UPTKEglFrameTypeTocudaEglFrameType(UPTKEglFrameType para) {
//     switch (para) {
//         case UPTKEglFrameTypeArray:
//             return cudaEglFrameTypeArray;
//         case UPTKEglFrameTypePitch:
//             return cudaEglFrameTypePitch;
//         default:
//             ERROR_INVALID_ENUM();
//     }
// }

// enum cudaEglResourceLocationFlags UPTKEglResourceLocationFlagsTocudaEglResourceLocationFlags(UPTKEglResourceLocationFlags para) {
//     switch (para) {
//         case UPTKEglResourceLocationSysmem:
//             return cudaEglResourceLocationSysmem;
//         case UPTKEglResourceLocationVidmem:
//             return cudaEglResourceLocationVidmem;
//         default:
//             ERROR_INVALID_ENUM();
//     }
// }

cudaError_t UPTKErrorTocudaError(enum UPTKError para) {
    switch (para) {
//         case UPTKErrorAddressOfConstant:
//             return cudaErrorAddressOfConstant;
        case UPTKErrorAlreadyAcquired:
            return cudaErrorAlreadyAcquired;
        case UPTKErrorAlreadyMapped:
            return cudaErrorAlreadyMapped;
//         case UPTKErrorApiFailureBase:
//             return cudaErrorApiFailureBase;
        case UPTKErrorArrayIsMapped:
            return cudaErrorArrayIsMapped;
        case UPTKErrorAssert:
            return cudaErrorAssert;
        case UPTKErrorCapturedEvent:
            return cudaErrorCapturedEvent;
//         case UPTKErrorCompatNotSupportedOnDevice:
//             return cudaErrorCompatNotSupportedOnDevice;
        case UPTKErrorContextIsDestroyed:
            return cudaErrorContextIsDestroyed;
        case UPTKErrorCooperativeLaunchTooLarge:
            return cudaErrorCooperativeLaunchTooLarge;
        case UPTKErrorCudartUnloading:
            return cudaErrorCudartUnloading;
        case UPTKErrorDeviceAlreadyInUse:
            return cudaErrorDeviceAlreadyInUse;
        case UPTKErrorDeviceUninitialized:
            return cudaErrorDeviceUninitialized;
//         case UPTKErrorDevicesUnavailable:
//             return cudaErrorDevicesUnavailable;
//         case UPTKErrorDuplicateSurfaceName:
//             return cudaErrorDuplicateSurfaceName;
//         case UPTKErrorDuplicateTextureName:
//             return cudaErrorDuplicateTextureName;
//         case UPTKErrorDuplicateVariableName:
//             return cudaErrorDuplicateVariableName;
        case UPTKErrorECCUncorrectable:
            return cudaErrorECCUncorrectable;
        case UPTKErrorFileNotFound:
            return cudaErrorFileNotFound;
        case UPTKErrorGraphExecUpdateFailure:
            return cudaErrorGraphExecUpdateFailure;
//         case UPTKErrorHardwareStackError:
//             return cudaErrorHardwareStackError;
        case UPTKErrorHostMemoryAlreadyRegistered:
            return cudaErrorHostMemoryAlreadyRegistered;
        case UPTKErrorHostMemoryNotRegistered:
            return cudaErrorHostMemoryNotRegistered;
        case UPTKErrorIllegalAddress:
            return cudaErrorIllegalAddress;
//         case UPTKErrorIllegalInstruction:
//             return cudaErrorIllegalInstruction;
        case UPTKErrorIllegalState:
            return cudaErrorIllegalState;
//         case UPTKErrorIncompatibleDriverContext:
//             return cudaErrorIncompatibleDriverContext;
        case UPTKErrorInitializationError:
            return cudaErrorInitializationError;
        case UPTKErrorInsufficientDriver:
            return cudaErrorInsufficientDriver;
//         case UPTKErrorInvalidAddressSpace:
//             return cudaErrorInvalidAddressSpace;
//         case UPTKErrorInvalidChannelDescriptor:
//             return cudaErrorInvalidChannelDescriptor;
        case UPTKErrorInvalidConfiguration:
            return cudaErrorInvalidConfiguration;
        case UPTKErrorInvalidDevice:
            return cudaErrorInvalidDevice;
        case UPTKErrorInvalidDeviceFunction:
            return cudaErrorInvalidDeviceFunction;
        case UPTKErrorInvalidDevicePointer:
            return cudaErrorInvalidDevicePointer;
//         case UPTKErrorInvalidFilterSetting:
//             return cudaErrorInvalidFilterSetting;
        case UPTKErrorInvalidGraphicsContext:
            return cudaErrorInvalidGraphicsContext;
//         case UPTKErrorInvalidHostPointer:
//             return cudaErrorInvalidHostPointer;
        case UPTKErrorInvalidKernelImage:
            return cudaErrorInvalidKernelImage;
        case UPTKErrorInvalidMemcpyDirection:
            return cudaErrorInvalidMemcpyDirection;
//         case UPTKErrorInvalidNormSetting:
//             return cudaErrorInvalidNormSetting;
//         case UPTKErrorInvalidPc:
//             return cudaErrorInvalidPc;
        case UPTKErrorInvalidPitchValue:
            return cudaErrorInvalidPitchValue;
        case UPTKErrorInvalidPtx:
            return cudaErrorInvalidPtx;
        case UPTKErrorInvalidResourceHandle:
            return cudaErrorInvalidResourceHandle;
        case UPTKErrorInvalidSource:
            return cudaErrorInvalidSource;
//         case UPTKErrorInvalidSurface:
//             return cudaErrorInvalidSurface;
        case UPTKErrorInvalidSymbol:
            return cudaErrorInvalidSymbol;
//         case UPTKErrorInvalidTexture:
//             return cudaErrorInvalidTexture;
//         case UPTKErrorInvalidTextureBinding:
//             return cudaErrorInvalidTextureBinding;
        case UPTKErrorInvalidValue:
            return cudaErrorInvalidValue;
//         case UPTKErrorJitCompilerNotFound:
//             return cudaErrorJitCompilerNotFound;
        case UPTKErrorLaunchFailure:
            return cudaErrorLaunchFailure;
//         case UPTKErrorLaunchFileScopedSurf:
//             return cudaErrorLaunchFileScopedSurf;
//         case UPTKErrorLaunchFileScopedTex:
//             return cudaErrorLaunchFileScopedTex;
//         case UPTKErrorLaunchIncompatibleTexturing:
//             return cudaErrorLaunchIncompatibleTexturing;
//         case UPTKErrorLaunchMaxDepthExceeded:
//             return cudaErrorLaunchMaxDepthExceeded;
        case UPTKErrorLaunchOutOfResources:
            return cudaErrorLaunchOutOfResources;
//         case UPTKErrorLaunchPendingCountExceeded:
//             return cudaErrorLaunchPendingCountExceeded;
        case UPTKErrorLaunchTimeout:
            return cudaErrorLaunchTimeout;
        case UPTKErrorMapBufferObjectFailed:
            return cudaErrorMapBufferObjectFailed;
        case UPTKErrorMemoryAllocation:
            return cudaErrorMemoryAllocation;
//         case UPTKErrorMemoryValueTooLarge:
//             return cudaErrorMemoryValueTooLarge;
//         case UPTKErrorMisalignedAddress:
//             return cudaErrorMisalignedAddress;
        case UPTKErrorMissingConfiguration:
            return cudaErrorMissingConfiguration;
//         case UPTKErrorMixedDeviceExecution:
//             return cudaErrorMixedDeviceExecution;
        case UPTKErrorNoDevice:
            return cudaErrorNoDevice;
        case UPTKErrorNoKernelImageForDevice:
            return cudaErrorNoKernelImageForDevice;
        case UPTKErrorNotMapped:
            return cudaErrorNotMapped;
        case UPTKErrorNotMappedAsArray:
            return cudaErrorNotMappedAsArray;
        case UPTKErrorNotMappedAsPointer:
            return cudaErrorNotMappedAsPointer;
//         case UPTKErrorNotPermitted:
//             return cudaErrorNotPermitted;
        case UPTKErrorNotReady:
            return cudaErrorNotReady;
        case UPTKErrorNotSupported:
            return cudaErrorNotSupported;
//         case UPTKErrorNotYetImplemented:
//             return cudaErrorNotYetImplemented;
//         case UPTKErrorNvlinkUncorrectable:
//             return cudaErrorNvlinkUncorrectable;
        case UPTKErrorOperatingSystem:
            return cudaErrorOperatingSystem;
        case UPTKErrorPeerAccessAlreadyEnabled:
            return cudaErrorPeerAccessAlreadyEnabled;
        case UPTKErrorPeerAccessNotEnabled:
            return cudaErrorPeerAccessNotEnabled;
        case UPTKErrorPeerAccessUnsupported:
            return cudaErrorPeerAccessUnsupported;
        case UPTKErrorPriorLaunchFailure:
            return cudaErrorPriorLaunchFailure;
        case UPTKErrorProfilerAlreadyStarted:
            return cudaErrorProfilerAlreadyStarted;
        case UPTKErrorProfilerAlreadyStopped:
            return cudaErrorProfilerAlreadyStopped;
        case UPTKErrorProfilerDisabled:
            return cudaErrorProfilerDisabled;
        case UPTKErrorProfilerNotInitialized:
            return cudaErrorProfilerNotInitialized;
        case UPTKErrorSetOnActiveProcess:
            return cudaErrorSetOnActiveProcess;
        case UPTKErrorSharedObjectInitFailed:
            return cudaErrorSharedObjectInitFailed;
        case UPTKErrorSharedObjectSymbolNotFound:
            return cudaErrorSharedObjectSymbolNotFound;
//         case UPTKErrorStartupFailure:
//             return cudaErrorStartupFailure;
        case UPTKErrorStreamCaptureImplicit:
            return cudaErrorStreamCaptureImplicit;
        case UPTKErrorStreamCaptureInvalidated:
            return cudaErrorStreamCaptureInvalidated;
        case UPTKErrorStreamCaptureIsolation:
            return cudaErrorStreamCaptureIsolation;
        case UPTKErrorStreamCaptureMerge:
            return cudaErrorStreamCaptureMerge;
        case UPTKErrorStreamCaptureUnjoined:
            return cudaErrorStreamCaptureUnjoined;
        case UPTKErrorStreamCaptureUnmatched:
            return cudaErrorStreamCaptureUnmatched;
        case UPTKErrorStreamCaptureUnsupported:
            return cudaErrorStreamCaptureUnsupported;
        case UPTKErrorStreamCaptureWrongThread:
            return cudaErrorStreamCaptureWrongThread;
        case UPTKErrorSymbolNotFound:
            return cudaErrorSymbolNotFound;
//         case UPTKErrorSyncDepthExceeded:
//             return cudaErrorSyncDepthExceeded;
//         case UPTKErrorSynchronizationError:
//             return cudaErrorSynchronizationError;
//         case UPTKErrorSystemDriverMismatch:
//             return cudaErrorSystemDriverMismatch;
//         case UPTKErrorSystemNotReady:
//             return cudaErrorSystemNotReady;
//         case UPTKErrorTextureFetchFailed:
//             return cudaErrorTextureFetchFailed;
//         case UPTKErrorTextureNotBound:
//             return cudaErrorTextureNotBound;
//         case UPTKErrorTimeout:
//             return cudaErrorTimeout;
//         case UPTKErrorTooManyPeers:
//             return cudaErrorTooManyPeers;
        case UPTKErrorUnknown:
            return cudaErrorUnknown;
        case UPTKErrorUnmapBufferObjectFailed:
            return cudaErrorUnmapBufferObjectFailed;
        case UPTKErrorUnsupportedLimit:
            return cudaErrorUnsupportedLimit;
        case UPTKErrorInvalidResourceType:
            return cudaErrorInvalidResourceType;
        // UPTK12.6 Added error codes do not correspond to cuda error codes
        // case UPTKErrorFunctionNotLoaded:
        //     return ;      
        // case UPTKErrorInvalidResourceConfiguration:
        //     return ;        
        case UPTKSuccess:
            return cudaSuccess;
        default:
            ERROR_INVALID_ENUM();
    }
}

enum UPTKError cudaErrorToUPTKError(cudaError_t para) {
    switch (para) {
//         case cudaErrorAddressOfConstant:
//             return UPTKErrorAddressOfConstant;
        case cudaErrorAlreadyAcquired:
            return UPTKErrorAlreadyAcquired;
        case cudaErrorAlreadyMapped:
            return UPTKErrorAlreadyMapped;
//         case cudaErrorApiFailureBase:
//             return UPTKErrorApiFailureBase;
        case cudaErrorArrayIsMapped:
            return UPTKErrorArrayIsMapped;
        case cudaErrorAssert:
            return UPTKErrorAssert;
        case cudaErrorCapturedEvent:
            return UPTKErrorCapturedEvent;
//         case cudaErrorCompatNotSupportedOnDevice:
//             return UPTKErrorCompatNotSupportedOnDevice;
        case cudaErrorContextIsDestroyed:
            return UPTKErrorContextIsDestroyed;
        case cudaErrorCooperativeLaunchTooLarge:
            return UPTKErrorCooperativeLaunchTooLarge;
        case cudaErrorCudartUnloading:
            return UPTKErrorCudartUnloading;
        case cudaErrorDeviceAlreadyInUse:
            return UPTKErrorDeviceAlreadyInUse;
        case cudaErrorDeviceUninitialized:
            return UPTKErrorDeviceUninitialized;
//         case cudaErrorDevicesUnavailable:
//             return UPTKErrorDevicesUnavailable;
//         case cudaErrorDuplicateSurfaceName:
//             return UPTKErrorDuplicateSurfaceName;
//         case cudaErrorDuplicateTextureName:
//             return UPTKErrorDuplicateTextureName;
//         case cudaErrorDuplicateVariableName:
//             return UPTKErrorDuplicateVariableName;
        case cudaErrorECCUncorrectable:
            return UPTKErrorECCUncorrectable;
        case cudaErrorFileNotFound:
            return UPTKErrorFileNotFound;
        case cudaErrorGraphExecUpdateFailure:
            return UPTKErrorGraphExecUpdateFailure;
//         case cudaErrorHardwareStackError:
//             return UPTKErrorHardwareStackError;
        case cudaErrorHostMemoryAlreadyRegistered:
            return UPTKErrorHostMemoryAlreadyRegistered;
        case cudaErrorHostMemoryNotRegistered:
            return UPTKErrorHostMemoryNotRegistered;
        case cudaErrorIllegalAddress:
            return UPTKErrorIllegalAddress;
//         case cudaErrorIllegalInstruction:
//             return UPTKErrorIllegalInstruction;
        case cudaErrorIllegalState:
            return UPTKErrorIllegalState;
//         case cudaErrorIncompatibleDriverContext:
//             return UPTKErrorIncompatibleDriverContext;
        case cudaErrorInitializationError:
            return UPTKErrorInitializationError;
        case cudaErrorInsufficientDriver:
            return UPTKErrorInsufficientDriver;
//         case cudaErrorInvalidAddressSpace:
//             return UPTKErrorInvalidAddressSpace;
//         case cudaErrorInvalidChannelDescriptor:
//             return UPTKErrorInvalidChannelDescriptor;
        case cudaErrorInvalidConfiguration:
            return UPTKErrorInvalidConfiguration;
        case cudaErrorInvalidDevice:
            return UPTKErrorInvalidDevice;
        case cudaErrorInvalidDeviceFunction:
            return UPTKErrorInvalidDeviceFunction;
        case cudaErrorInvalidDevicePointer:
            return UPTKErrorInvalidDevicePointer;
//         case cudaErrorInvalidFilterSetting:
//             return UPTKErrorInvalidFilterSetting;
        case cudaErrorInvalidGraphicsContext:
            return UPTKErrorInvalidGraphicsContext;
//         case cudaErrorInvalidHostPointer:
//             return UPTKErrorInvalidHostPointer;
        case cudaErrorInvalidKernelImage:
            return UPTKErrorInvalidKernelImage;
        case cudaErrorInvalidMemcpyDirection:
            return UPTKErrorInvalidMemcpyDirection;
//         case cudaErrorInvalidNormSetting:
//             return UPTKErrorInvalidNormSetting;
//         case cudaErrorInvalidPc:
//             return UPTKErrorInvalidPc;
        case cudaErrorInvalidPitchValue:
            return UPTKErrorInvalidPitchValue;
        case cudaErrorInvalidPtx:
            return UPTKErrorInvalidPtx;
        case cudaErrorInvalidResourceHandle:
            return UPTKErrorInvalidResourceHandle;
        case cudaErrorInvalidSource:
            return UPTKErrorInvalidSource;
//         case cudaErrorInvalidSurface:
//             return UPTKErrorInvalidSurface;
        case cudaErrorInvalidSymbol:
            return UPTKErrorInvalidSymbol;
//         case cudaErrorInvalidTexture:
//             return UPTKErrorInvalidTexture;
//         case cudaErrorInvalidTextureBinding:
//             return UPTKErrorInvalidTextureBinding;
        case cudaErrorInvalidValue:
            return UPTKErrorInvalidValue;
//         case cudaErrorJitCompilerNotFound:
//             return UPTKErrorJitCompilerNotFound;
        case cudaErrorLaunchFailure:
            return UPTKErrorLaunchFailure;
//         case cudaErrorLaunchFileScopedSurf:
//             return UPTKErrorLaunchFileScopedSurf;
//         case cudaErrorLaunchFileScopedTex:
//             return UPTKErrorLaunchFileScopedTex;
//         case cudaErrorLaunchIncompatibleTexturing:
//             return UPTKErrorLaunchIncompatibleTexturing;
//         case cudaErrorLaunchMaxDepthExceeded:
//             return UPTKErrorLaunchMaxDepthExceeded;
        case cudaErrorLaunchOutOfResources:
            return UPTKErrorLaunchOutOfResources;
//         case cudaErrorLaunchPendingCountExceeded:
//             return UPTKErrorLaunchPendingCountExceeded;
        case cudaErrorLaunchTimeout:
            return UPTKErrorLaunchTimeout;
        case cudaErrorMapBufferObjectFailed:
            return UPTKErrorMapBufferObjectFailed;
        case cudaErrorMemoryAllocation:
            return UPTKErrorMemoryAllocation;
//         case cudaErrorMemoryValueTooLarge:
//             return UPTKErrorMemoryValueTooLarge;
//         case cudaErrorMisalignedAddress:
//             return UPTKErrorMisalignedAddress;
        case cudaErrorMissingConfiguration:
            return UPTKErrorMissingConfiguration;
//         case cudaErrorMixedDeviceExecution:
//             return UPTKErrorMixedDeviceExecution;
        case cudaErrorNoDevice:
            return UPTKErrorNoDevice;
        case cudaErrorNoKernelImageForDevice:
            return UPTKErrorNoKernelImageForDevice;
        case cudaErrorNotMapped:
            return UPTKErrorNotMapped;
        case cudaErrorNotMappedAsArray:
            return UPTKErrorNotMappedAsArray;
        case cudaErrorNotMappedAsPointer:
            return UPTKErrorNotMappedAsPointer;
//         case cudaErrorNotPermitted:
//             return UPTKErrorNotPermitted;
        case cudaErrorNotReady:
            return UPTKErrorNotReady;
        case cudaErrorNotSupported:
            return UPTKErrorNotSupported;
//         case cudaErrorNotYetImplemented:
//             return UPTKErrorNotYetImplemented;
//         case cudaErrorNvlinkUncorrectable:
//             return UPTKErrorNvlinkUncorrectable;
        case cudaErrorOperatingSystem:
            return UPTKErrorOperatingSystem;
        case cudaErrorPeerAccessAlreadyEnabled:
            return UPTKErrorPeerAccessAlreadyEnabled;
        case cudaErrorPeerAccessNotEnabled:
            return UPTKErrorPeerAccessNotEnabled;
        case cudaErrorPeerAccessUnsupported:
            return UPTKErrorPeerAccessUnsupported;
        case cudaErrorPriorLaunchFailure:
            return UPTKErrorPriorLaunchFailure;
        case cudaErrorProfilerAlreadyStarted:
            return UPTKErrorProfilerAlreadyStarted;
        case cudaErrorProfilerAlreadyStopped:
            return UPTKErrorProfilerAlreadyStopped;
        case cudaErrorProfilerDisabled:
            return UPTKErrorProfilerDisabled;
        case cudaErrorProfilerNotInitialized:
            return UPTKErrorProfilerNotInitialized;
        case cudaErrorSetOnActiveProcess:
            return UPTKErrorSetOnActiveProcess;
        case cudaErrorSharedObjectInitFailed:
            return UPTKErrorSharedObjectInitFailed;
        case cudaErrorSharedObjectSymbolNotFound:
            return UPTKErrorSharedObjectSymbolNotFound;
//         case cudaErrorStartupFailure:
//             return UPTKErrorStartupFailure;
        case cudaErrorStreamCaptureImplicit:
            return UPTKErrorStreamCaptureImplicit;
        case cudaErrorStreamCaptureInvalidated:
            return UPTKErrorStreamCaptureInvalidated;
        case cudaErrorStreamCaptureIsolation:
            return UPTKErrorStreamCaptureIsolation;
        case cudaErrorStreamCaptureMerge:
            return UPTKErrorStreamCaptureMerge;
        case cudaErrorStreamCaptureUnjoined:
            return UPTKErrorStreamCaptureUnjoined;
        case cudaErrorStreamCaptureUnmatched:
            return UPTKErrorStreamCaptureUnmatched;
        case cudaErrorStreamCaptureUnsupported:
            return UPTKErrorStreamCaptureUnsupported;
        case cudaErrorStreamCaptureWrongThread:
            return UPTKErrorStreamCaptureWrongThread;
        case cudaErrorSymbolNotFound:
            return UPTKErrorSymbolNotFound;
//         case cudaErrorSyncDepthExceeded:
//             return UPTKErrorSyncDepthExceeded;
//         case cudaErrorSynchronizationError:
//             return UPTKErrorSynchronizationError;
//         case cudaErrorSystemDriverMismatch:
//             return UPTKErrorSystemDriverMismatch;
//         case cudaErrorSystemNotReady:
//             return UPTKErrorSystemNotReady;
//         case cudaErrorTextureFetchFailed:
//             return UPTKErrorTextureFetchFailed;
//         case cudaErrorTextureNotBound:
//             return UPTKErrorTextureNotBound;
//         case cudaErrorTimeout:
//             return UPTKErrorTimeout;
//         case cudaErrorTooManyPeers:
//             return UPTKErrorTooManyPeers;
        case cudaErrorUnknown:
            return UPTKErrorUnknown;
        case cudaErrorUnmapBufferObjectFailed:
            return UPTKErrorUnmapBufferObjectFailed;
        case cudaErrorUnsupportedLimit:
            return UPTKErrorUnsupportedLimit;
        case cudaErrorInvalidResourceType:
            return UPTKErrorInvalidResourceType;    
        case cudaSuccess:
            return UPTKSuccess;
        default:
            ERROR_INVALID_ENUM();
    }
}
// In a non-DCU environment, the UPTKGetDeviceProperties interface error code is aligned with the NV
enum UPTKError cudaErrorToUPTKError_v2(cudaError_t para) {
    switch (para) {
//         case cudaErrorAddressOfConstant:
//             return UPTKErrorAddressOfConstant;
        case cudaErrorAlreadyAcquired:
            return UPTKErrorAlreadyAcquired;
        case cudaErrorAlreadyMapped:
            return UPTKErrorAlreadyMapped;
//         case cudaErrorApiFailureBase:
//             return UPTKErrorApiFailureBase;
        case cudaErrorArrayIsMapped:
            return UPTKErrorArrayIsMapped;
        case cudaErrorAssert:
            return UPTKErrorAssert;
        case cudaErrorCapturedEvent:
            return UPTKErrorCapturedEvent;
//         case cudaErrorCompatNotSupportedOnDevice:
//             return UPTKErrorCompatNotSupportedOnDevice;
        case cudaErrorContextIsDestroyed:
            return UPTKErrorContextIsDestroyed;
        case cudaErrorCooperativeLaunchTooLarge:
            return UPTKErrorCooperativeLaunchTooLarge;
        case cudaErrorCudartUnloading:
            return UPTKErrorCudartUnloading;
        case cudaErrorDeviceAlreadyInUse:
            return UPTKErrorDeviceAlreadyInUse;
        case cudaErrorDeviceUninitialized:
            return UPTKErrorDeviceUninitialized;
//         case cudaErrorDevicesUnavailable:
//             return UPTKErrorDevicesUnavailable;
//         case cudaErrorDuplicateSurfaceName:
//             return UPTKErrorDuplicateSurfaceName;
//         case cudaErrorDuplicateTextureName:
//             return UPTKErrorDuplicateTextureName;
//         case cudaErrorDuplicateVariableName:
//             return UPTKErrorDuplicateVariableName;
        case cudaErrorECCUncorrectable:
            return UPTKErrorECCUncorrectable;
        case cudaErrorFileNotFound:
            return UPTKErrorFileNotFound;
        case cudaErrorGraphExecUpdateFailure:
            return UPTKErrorGraphExecUpdateFailure;
//         case cudaErrorHardwareStackError:
//             return UPTKErrorHardwareStackError;
        case cudaErrorHostMemoryAlreadyRegistered:
            return UPTKErrorHostMemoryAlreadyRegistered;
        case cudaErrorHostMemoryNotRegistered:
            return UPTKErrorHostMemoryNotRegistered;
        case cudaErrorIllegalAddress:
            return UPTKErrorIllegalAddress;
//         case cudaErrorIllegalInstruction:
//             return UPTKErrorIllegalInstruction;
        case cudaErrorIllegalState:
            return UPTKErrorIllegalState;
//         case cudaErrorIncompatibleDriverContext:
//             return UPTKErrorIncompatibleDriverContext;
        case cudaErrorInitializationError:
            return UPTKErrorInitializationError;
        case cudaErrorInsufficientDriver:
            return UPTKErrorInsufficientDriver;
//         case cudaErrorInvalidAddressSpace:
//             return UPTKErrorInvalidAddressSpace;
//         case cudaErrorInvalidChannelDescriptor:
//             return UPTKErrorInvalidChannelDescriptor;
        case cudaErrorInvalidConfiguration:
            return UPTKErrorInvalidConfiguration;
        // case cudaErrorInvalidDevice:
        //     return UPTKErrorInvalidDevice;
        // Alignment differences: no dcu card environment, call UPTKGetDeviceProperties interface, 
        // the official return to UPTKErrorInsufficientDriver NV, return to UPTKErrorInvalidDevice dcu
        case cudaErrorInvalidDeviceFunction:
            return UPTKErrorInvalidDeviceFunction;
        case cudaErrorInvalidDevicePointer:
            return UPTKErrorInvalidDevicePointer;
//         case cudaErrorInvalidFilterSetting:
//             return UPTKErrorInvalidFilterSetting;
        case cudaErrorInvalidGraphicsContext:
            return UPTKErrorInvalidGraphicsContext;
//         case cudaErrorInvalidHostPointer:
//             return UPTKErrorInvalidHostPointer;
        case cudaErrorInvalidKernelImage:
            return UPTKErrorInvalidKernelImage;
        case cudaErrorInvalidMemcpyDirection:
            return UPTKErrorInvalidMemcpyDirection;
//         case cudaErrorInvalidNormSetting:
//             return UPTKErrorInvalidNormSetting;
//         case cudaErrorInvalidPc:
//             return UPTKErrorInvalidPc;
        case cudaErrorInvalidPitchValue:
            return UPTKErrorInvalidPitchValue;
        case cudaErrorInvalidPtx:
            return UPTKErrorInvalidPtx;
        case cudaErrorInvalidResourceHandle:
            return UPTKErrorInvalidResourceHandle;
        case cudaErrorInvalidSource:
            return UPTKErrorInvalidSource;
//         case cudaErrorInvalidSurface:
//             return UPTKErrorInvalidSurface;
        case cudaErrorInvalidSymbol:
            return UPTKErrorInvalidSymbol;
//         case cudaErrorInvalidTexture:
//             return UPTKErrorInvalidTexture;
//         case cudaErrorInvalidTextureBinding:
//             return UPTKErrorInvalidTextureBinding;
        case cudaErrorInvalidValue:
            return UPTKErrorInvalidValue;
//         case cudaErrorJitCompilerNotFound:
//             return UPTKErrorJitCompilerNotFound;
        case cudaErrorLaunchFailure:
            return UPTKErrorLaunchFailure;
//         case cudaErrorLaunchFileScopedSurf:
//             return UPTKErrorLaunchFileScopedSurf;
//         case cudaErrorLaunchFileScopedTex:
//             return UPTKErrorLaunchFileScopedTex;
//         case cudaErrorLaunchIncompatibleTexturing:
//             return UPTKErrorLaunchIncompatibleTexturing;
//         case cudaErrorLaunchMaxDepthExceeded:
//             return UPTKErrorLaunchMaxDepthExceeded;
        case cudaErrorLaunchOutOfResources:
            return UPTKErrorLaunchOutOfResources;
//         case cudaErrorLaunchPendingCountExceeded:
//             return UPTKErrorLaunchPendingCountExceeded;
        case cudaErrorLaunchTimeout:
            return UPTKErrorLaunchTimeout;
        case cudaErrorMapBufferObjectFailed:
            return UPTKErrorMapBufferObjectFailed;
        case cudaErrorMemoryAllocation:
            return UPTKErrorMemoryAllocation;
//         case cudaErrorMemoryValueTooLarge:
//             return UPTKErrorMemoryValueTooLarge;
//         case cudaErrorMisalignedAddress:
//             return UPTKErrorMisalignedAddress;
        case cudaErrorMissingConfiguration:
            return UPTKErrorMissingConfiguration;
//         case cudaErrorMixedDeviceExecution:
//             return UPTKErrorMixedDeviceExecution;
        case cudaErrorNoDevice:
            return UPTKErrorNoDevice;
        case cudaErrorNoKernelImageForDevice:
            return UPTKErrorNoKernelImageForDevice;
        case cudaErrorNotMapped:
            return UPTKErrorNotMapped;
        case cudaErrorNotMappedAsArray:
            return UPTKErrorNotMappedAsArray;
        case cudaErrorNotMappedAsPointer:
            return UPTKErrorNotMappedAsPointer;
//         case cudaErrorNotPermitted:
//             return UPTKErrorNotPermitted;
        case cudaErrorNotReady:
            return UPTKErrorNotReady;
        case cudaErrorNotSupported:
            return UPTKErrorNotSupported;
//         case cudaErrorNotYetImplemented:
//             return UPTKErrorNotYetImplemented;
//         case cudaErrorNvlinkUncorrectable:
//             return UPTKErrorNvlinkUncorrectable;
        case cudaErrorOperatingSystem:
            return UPTKErrorOperatingSystem;
        case cudaErrorPeerAccessAlreadyEnabled:
            return UPTKErrorPeerAccessAlreadyEnabled;
        case cudaErrorPeerAccessNotEnabled:
            return UPTKErrorPeerAccessNotEnabled;
        case cudaErrorPeerAccessUnsupported:
            return UPTKErrorPeerAccessUnsupported;
        case cudaErrorPriorLaunchFailure:
            return UPTKErrorPriorLaunchFailure;
        case cudaErrorProfilerAlreadyStarted:
            return UPTKErrorProfilerAlreadyStarted;
        case cudaErrorProfilerAlreadyStopped:
            return UPTKErrorProfilerAlreadyStopped;
        case cudaErrorProfilerDisabled:
            return UPTKErrorProfilerDisabled;
        case cudaErrorProfilerNotInitialized:
            return UPTKErrorProfilerNotInitialized;
        case cudaErrorSetOnActiveProcess:
            return UPTKErrorSetOnActiveProcess;
        case cudaErrorSharedObjectInitFailed:
            return UPTKErrorSharedObjectInitFailed;
        case cudaErrorSharedObjectSymbolNotFound:
            return UPTKErrorSharedObjectSymbolNotFound;
//         case cudaErrorStartupFailure:
//             return UPTKErrorStartupFailure;
        case cudaErrorStreamCaptureImplicit:
            return UPTKErrorStreamCaptureImplicit;
        case cudaErrorStreamCaptureInvalidated:
            return UPTKErrorStreamCaptureInvalidated;
        case cudaErrorStreamCaptureIsolation:
            return UPTKErrorStreamCaptureIsolation;
        case cudaErrorStreamCaptureMerge:
            return UPTKErrorStreamCaptureMerge;
        case cudaErrorStreamCaptureUnjoined:
            return UPTKErrorStreamCaptureUnjoined;
        case cudaErrorStreamCaptureUnmatched:
            return UPTKErrorStreamCaptureUnmatched;
        case cudaErrorStreamCaptureUnsupported:
            return UPTKErrorStreamCaptureUnsupported;
        case cudaErrorStreamCaptureWrongThread:
            return UPTKErrorStreamCaptureWrongThread;
        case cudaErrorSymbolNotFound:
            return UPTKErrorSymbolNotFound;
//         case cudaErrorSyncDepthExceeded:
//             return UPTKErrorSyncDepthExceeded;
//         case cudaErrorSynchronizationError:
//             return UPTKErrorSynchronizationError;
//         case cudaErrorSystemDriverMismatch:
//             return UPTKErrorSystemDriverMismatch;
//         case cudaErrorSystemNotReady:
//             return UPTKErrorSystemNotReady;
//         case cudaErrorTextureFetchFailed:
//             return UPTKErrorTextureFetchFailed;
//         case cudaErrorTextureNotBound:
//             return UPTKErrorTextureNotBound;
//         case cudaErrorTimeout:
//             return UPTKErrorTimeout;
//         case cudaErrorTooManyPeers:
//             return UPTKErrorTooManyPeers;
        case cudaErrorUnknown:
            return UPTKErrorUnknown;
        case cudaErrorUnmapBufferObjectFailed:
            return UPTKErrorUnmapBufferObjectFailed;
        case cudaErrorUnsupportedLimit:
            return UPTKErrorUnsupportedLimit;
        case cudaErrorInvalidResourceType:
            return UPTKErrorInvalidResourceType;        
        case cudaSuccess:
            return UPTKSuccess;
        default:
            ERROR_INVALID_ENUM();
    }
}

cudaExternalMemoryHandleType UPTKExternalMemoryHandleTypeTocudaExternalMemoryHandleType(enum UPTKExternalMemoryHandleType para) {
    switch (para) {
        case UPTKExternalMemoryHandleTypeD3D11Resource:
            return cudaExternalMemoryHandleTypeD3D11Resource;
        case UPTKExternalMemoryHandleTypeD3D11ResourceKmt:
            return cudaExternalMemoryHandleTypeD3D11ResourceKmt;
        case UPTKExternalMemoryHandleTypeD3D12Heap:
            return cudaExternalMemoryHandleTypeD3D12Heap;
        case UPTKExternalMemoryHandleTypeD3D12Resource:
            return cudaExternalMemoryHandleTypeD3D12Resource;
//         case UPTKExternalMemoryHandleTypeNvSciBuf:
//             return cudaExternalMemoryHandleTypeNvSciBuf;
        case UPTKExternalMemoryHandleTypeOpaqueFd:
            return cudaExternalMemoryHandleTypeOpaqueFd;
        case UPTKExternalMemoryHandleTypeOpaqueWin32:
            return cudaExternalMemoryHandleTypeOpaqueWin32;
        case UPTKExternalMemoryHandleTypeOpaqueWin32Kmt:
            return cudaExternalMemoryHandleTypeOpaqueWin32Kmt;
        default:
            ERROR_INVALID_ENUM();
    }
}

cudaExternalSemaphoreHandleType UPTKExternalSemaphoreHandleTypeTocudaExternalSemaphoreHandleType(enum UPTKExternalSemaphoreHandleType para) {
    switch (para) {
//         case UPTKExternalSemaphoreHandleTypeD3D11Fence:
//             return cudaExternalSemaphoreHandleTypeD3D11Fence;
        case UPTKExternalSemaphoreHandleTypeD3D12Fence:
            return cudaExternalSemaphoreHandleTypeD3D12Fence;
//         case UPTKExternalSemaphoreHandleTypeKeyedMutex:
//             return cudaExternalSemaphoreHandleTypeKeyedMutex;
//         case UPTKExternalSemaphoreHandleTypeKeyedMutexKmt:
//             return cudaExternalSemaphoreHandleTypeKeyedMutexKmt;
//         case UPTKExternalSemaphoreHandleTypeNvSciSync:
//             return cudaExternalSemaphoreHandleTypeNvSciSync;
        case UPTKExternalSemaphoreHandleTypeOpaqueFd:
            return cudaExternalSemaphoreHandleTypeOpaqueFd;
        case UPTKExternalSemaphoreHandleTypeOpaqueWin32:
            return cudaExternalSemaphoreHandleTypeOpaqueWin32;
        case UPTKExternalSemaphoreHandleTypeOpaqueWin32Kmt:
            return cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt;
        default:
            ERROR_INVALID_ENUM();
    }
}

enum cudaTextureFilterMode UPTKTextureFilterModeTocudaTextureFilterMode(enum UPTKTextureFilterMode para) {
    switch (para) {
        case UPTKFilterModeLinear:
            return cudaFilterModeLinear;
        case UPTKFilterModePoint:
            return cudaFilterModePoint;
        default:
            ERROR_INVALID_ENUM();
    }
}

// enum cudaSurfaceFormatMode UPTKSurfaceFormatModeTocudaSurfaceFormatMode(enum UPTKSurfaceFormatMode para) {
//     switch (para) {
//         case UPTKFormatModeAuto:
//             return cudaFormatModeAuto;
//         case UPTKFormatModeForced:
//             return cudaFormatModeForced;
//         default:
//             ERROR_INVALID_ENUM();
//     }
// }

cudaFuncAttribute UPTKFuncAttributeTocudaFuncAttribute(enum UPTKFuncAttribute para) {
    switch (para) {
        case UPTKFuncAttributeMax:
            return cudaFuncAttributeMax;
        case UPTKFuncAttributeMaxDynamicSharedMemorySize:
            return cudaFuncAttributeMaxDynamicSharedMemorySize;
        case UPTKFuncAttributePreferredSharedMemoryCarveout:
            return cudaFuncAttributePreferredSharedMemoryCarveout;
        default:
            ERROR_INVALID_ENUM();
    }
}

cudaFuncCache UPTKFuncCacheTocudaFuncCache(enum UPTKFuncCache para) {
    switch (para) {
        case UPTKFuncCachePreferEqual:
            return cudaFuncCachePreferEqual;
        case UPTKFuncCachePreferL1:
            return cudaFuncCachePreferL1;
        case UPTKFuncCachePreferNone:
            return cudaFuncCachePreferNone;
        case UPTKFuncCachePreferShared:
            return cudaFuncCachePreferShared;
        default:
            ERROR_INVALID_ENUM();
    }
}

enum UPTKFuncCache cudaFuncCacheToUPTKFuncCache(cudaFuncCache para) {
    switch (para) {
        case cudaFuncCachePreferEqual:
            return UPTKFuncCachePreferEqual;
        case cudaFuncCachePreferL1:
            return UPTKFuncCachePreferL1;
        case cudaFuncCachePreferNone:
            return UPTKFuncCachePreferNone;
        case cudaFuncCachePreferShared:
            return UPTKFuncCachePreferShared;
        default:
            ERROR_INVALID_ENUM();
    }
}

// enum cudaGLMapFlags UPTKGLMapFlagsTocudaGLMapFlags(enum UPTKGLMapFlags para) {
//     switch (para) {
//         case UPTKGLMapFlagsNone:
//             return cudaGLMapFlagsNone;
//         case UPTKGLMapFlagsReadOnly:
//             return cudaGLMapFlagsReadOnly;
//         case UPTKGLMapFlagsWriteDiscard:
//             return cudaGLMapFlagsWriteDiscard;
//         default:
//             ERROR_INVALID_ENUM();
//     }
// }

cudaGraphExecUpdateResult UPTKGraphExecUpdateResultTocudaGraphExecUpdateResult(enum UPTKGraphExecUpdateResult para) {
    switch (para) {
        case UPTKGraphExecUpdateError:
            return cudaGraphExecUpdateError;
        case UPTKGraphExecUpdateErrorFunctionChanged:
            return cudaGraphExecUpdateErrorFunctionChanged;
        case UPTKGraphExecUpdateErrorNodeTypeChanged:
            return cudaGraphExecUpdateErrorNodeTypeChanged;
        case UPTKGraphExecUpdateErrorNotSupported:
            return cudaGraphExecUpdateErrorNotSupported;
        case UPTKGraphExecUpdateErrorParametersChanged:
            return cudaGraphExecUpdateErrorParametersChanged;
        case UPTKGraphExecUpdateErrorTopologyChanged:
            return cudaGraphExecUpdateErrorTopologyChanged;
        case UPTKGraphExecUpdateErrorUnsupportedFunctionChange:
            return cudaGraphExecUpdateErrorUnsupportedFunctionChange;
        case UPTKGraphExecUpdateSuccess:
            return cudaGraphExecUpdateSuccess;
        default:
            ERROR_INVALID_ENUM();
    }
}

enum UPTKGraphExecUpdateResult cudaGraphExecUpdateResultToUPTKGraphExecUpdateResult(cudaGraphExecUpdateResult para) {
    switch (para) {
        case cudaGraphExecUpdateError:
            return UPTKGraphExecUpdateError;
        case cudaGraphExecUpdateErrorFunctionChanged:
            return UPTKGraphExecUpdateErrorFunctionChanged;
        case cudaGraphExecUpdateErrorNodeTypeChanged:
            return UPTKGraphExecUpdateErrorNodeTypeChanged;
        case cudaGraphExecUpdateErrorNotSupported:
            return UPTKGraphExecUpdateErrorNotSupported;
        case cudaGraphExecUpdateErrorParametersChanged:
            return UPTKGraphExecUpdateErrorParametersChanged;
        case cudaGraphExecUpdateErrorTopologyChanged:
            return UPTKGraphExecUpdateErrorTopologyChanged;
        case cudaGraphExecUpdateErrorUnsupportedFunctionChange:
            return UPTKGraphExecUpdateErrorUnsupportedFunctionChange;
        case cudaGraphExecUpdateSuccess:
            return UPTKGraphExecUpdateSuccess;
        default:
            ERROR_INVALID_ENUM();
    }
}
// UPTK 12.6：cudaGraphNodeType cudaGraphNodeTypeMemcpyFromSymbol, cudaGraphNodeTypeMemcpyToSymbol, cudaGraphNodeTypeBatchMemOp no UPTK corresponding members
cudaGraphNodeType UPTKGraphNodeTypeTocudaGraphNodeType(enum UPTKGraphNodeType para) {
    switch (para) {
        case UPTKGraphNodeTypeKernel:
            return cudaGraphNodeTypeKernel;
        case UPTKGraphNodeTypeMemcpy:
            return cudaGraphNodeTypeMemcpy;
        case UPTKGraphNodeTypeMemset:
            return cudaGraphNodeTypeMemset;
        case UPTKGraphNodeTypeHost:
            return cudaGraphNodeTypeHost;
        case UPTKGraphNodeTypeGraph:
            return cudaGraphNodeTypeGraph;
        case UPTKGraphNodeTypeEmpty:
            return cudaGraphNodeTypeEmpty;
        case UPTKGraphNodeTypeWaitEvent:
            return cudaGraphNodeTypeWaitEvent;
        case UPTKGraphNodeTypeEventRecord:
            return cudaGraphNodeTypeEventRecord;
        case UPTKGraphNodeTypeExtSemaphoreSignal:
            return cudaGraphNodeTypeExtSemaphoreSignal;
        case UPTKGraphNodeTypeExtSemaphoreWait:
            return cudaGraphNodeTypeExtSemaphoreWait;
        case UPTKGraphNodeTypeCount:
            return cudaGraphNodeTypeCount;
        case UPTKGraphNodeTypeMemAlloc:
            return cudaGraphNodeTypeMemAlloc;
        case UPTKGraphNodeTypeMemFree:
            return cudaGraphNodeTypeMemFree;
        case UPTKGraphNodeTypeConditional:
            return cudaGraphNodeTypeConditional;
        default: 
            ERROR_INVALID_ENUM();
    }
}

enum UPTKGraphNodeType cudaGraphNodeTypeToUPTKGraphNodeType(cudaGraphNodeType para) {
    switch (para) {
        case cudaGraphNodeTypeCount:
            return UPTKGraphNodeTypeCount;
        case cudaGraphNodeTypeEmpty:
            return UPTKGraphNodeTypeEmpty;
        case cudaGraphNodeTypeEventRecord:
            return UPTKGraphNodeTypeEventRecord;
        case cudaGraphNodeTypeGraph:
            return UPTKGraphNodeTypeGraph;
        case cudaGraphNodeTypeHost:
            return UPTKGraphNodeTypeHost;
        case cudaGraphNodeTypeKernel:
            return UPTKGraphNodeTypeKernel;
        case cudaGraphNodeTypeMemcpy:
            return UPTKGraphNodeTypeMemcpy;
        case cudaGraphNodeTypeMemset:
            return UPTKGraphNodeTypeMemset;
        case cudaGraphNodeTypeWaitEvent:
            return UPTKGraphNodeTypeWaitEvent;
        case cudaGraphNodeTypeExtSemaphoreSignal:
            return UPTKGraphNodeTypeExtSemaphoreSignal;
        case cudaGraphNodeTypeExtSemaphoreWait:
            return UPTKGraphNodeTypeExtSemaphoreWait;
        case cudaGraphNodeTypeMemAlloc:
            return UPTKGraphNodeTypeMemAlloc;
        case cudaGraphNodeTypeMemFree:
            return UPTKGraphNodeTypeMemFree;
        case cudaGraphNodeTypeConditional:
            return UPTKGraphNodeTypeConditional;
        default: 
            ERROR_INVALID_ENUM();
    }
}

// enum cudaGraphicsCubeFace UPTKGraphicsCubeFaceTocudaGraphicsCubeFace(enum UPTKGraphicsCubeFace para) {
//     switch (para) {
//         case UPTKGraphicsCubeFaceNegativeX:
//             return cudaGraphicsCubeFaceNegativeX;
//         case UPTKGraphicsCubeFaceNegativeY:
//             return cudaGraphicsCubeFaceNegativeY;
//         case UPTKGraphicsCubeFaceNegativeZ:
//             return cudaGraphicsCubeFaceNegativeZ;
//         case UPTKGraphicsCubeFacePositiveX:
//             return cudaGraphicsCubeFacePositiveX;
//         case UPTKGraphicsCubeFacePositiveY:
//             return cudaGraphicsCubeFacePositiveY;
//         case UPTKGraphicsCubeFacePositiveZ:
//             return cudaGraphicsCubeFacePositiveZ;
//         default:
//             ERROR_INVALID_ENUM();
//     }
// }

// enum cudaGraphicsMapFlags UPTKGraphicsMapFlagsTocudaGraphicsMapFlags(enum UPTKGraphicsMapFlags para) {
//     switch (para) {
//         case UPTKGraphicsMapFlagsNone:
//             return cudaGraphicsMapFlagsNone;
//         case UPTKGraphicsMapFlagsReadOnly:
//             return cudaGraphicsMapFlagsReadOnly;
//         case UPTKGraphicsMapFlagsWriteDiscard:
//             return cudaGraphicsMapFlagsWriteDiscard;
//         default:
//             ERROR_INVALID_ENUM();
//     }
// }

cudaGraphicsRegisterFlags UPTKGraphicsRegisterFlagsTocudaGraphicsRegisterFlags(enum UPTKGraphicsRegisterFlags para) {
    switch (para) {
        case UPTKGraphicsRegisterFlagsNone:
            return cudaGraphicsRegisterFlagsNone;
        case UPTKGraphicsRegisterFlagsReadOnly:
            return cudaGraphicsRegisterFlagsReadOnly;
        case UPTKGraphicsRegisterFlagsSurfaceLoadStore:
            return cudaGraphicsRegisterFlagsSurfaceLoadStore;
        case UPTKGraphicsRegisterFlagsTextureGather:
            return cudaGraphicsRegisterFlagsTextureGather;
        case UPTKGraphicsRegisterFlagsWriteDiscard:
            return cudaGraphicsRegisterFlagsWriteDiscard;
        default:
            ERROR_INVALID_ENUM();
    }
}

cudaKernelNodeAttrID UPTKKernelNodeAttrIDTocudaKernelNodeAttrID(enum UPTKKernelNodeAttrID para) {
    switch (para) {
        case UPTKKernelNodeAttributeAccessPolicyWindow:
            return cudaKernelNodeAttributeAccessPolicyWindow;
        case UPTKKernelNodeAttributeCooperative:
            return cudaKernelNodeAttributeCooperative;
        default:
            ERROR_INVALID_ENUM();
    }
}

enum cudaLimit UPTKLimitTocudaLimit(enum UPTKLimit para) {
    switch (para) {
        case UPTKLimitDevRuntimePendingLaunchCount:
            return cudaLimitDevRuntimePendingLaunchCount;
        case UPTKLimitDevRuntimeSyncDepth:
            return cudaLimitDevRuntimeSyncDepth;
        case UPTKLimitMallocHeapSize:
            return cudaLimitMallocHeapSize;
//         case UPTKLimitMaxL2FetchGranularity:
//             return cudaLimitMaxL2FetchGranularity;
        case UPTKLimitPrintfFifoSize:
            return cudaLimitPrintfFifoSize;
        case UPTKLimitStackSize:
            return cudaLimitStackSize;
        default:
            ERROR_INVALID_ENUM();
    }
}

cudaMemAccessFlags UPTKMemAccessFlagsTocudaMemAccessFlags(UPTKMemAccessFlags para) {
    switch (para) {
        case UPTKMemAccessFlagsProtNone:
            return cudaMemAccessFlagsProtNone;
        case UPTKMemAccessFlagsProtRead:
            return cudaMemAccessFlagsProtRead;
        case UPTKMemAccessFlagsProtReadWrite:
            return cudaMemAccessFlagsProtReadWrite;
        default:
            ERROR_INVALID_ENUM();
    }
}

UPTKMemAccessFlags cudaMemAccessFlagsToUPTKMemAccessFlags(cudaMemAccessFlags para) {
    switch (para) {
        case cudaMemAccessFlagsProtNone:
            return UPTKMemAccessFlagsProtNone;
        case cudaMemAccessFlagsProtRead:
            return UPTKMemAccessFlagsProtRead;
        case cudaMemAccessFlagsProtReadWrite:
            return UPTKMemAccessFlagsProtReadWrite;
        default:
            ERROR_INVALID_ENUM();
    }
}


cudaMemoryAdvise UPTKMemoryAdviseTocudaMemoryAdvise(enum UPTKMemoryAdvise para) {
    switch (para) {
        case UPTKMemAdviseSetAccessedBy:
            return cudaMemAdviseSetAccessedBy;
        case UPTKMemAdviseSetPreferredLocation:
            return cudaMemAdviseSetPreferredLocation;
        case UPTKMemAdviseSetReadMostly:
            return cudaMemAdviseSetReadMostly;
        case UPTKMemAdviseUnsetAccessedBy:
            return cudaMemAdviseUnsetAccessedBy;
        case UPTKMemAdviseUnsetPreferredLocation:
            return cudaMemAdviseUnsetPreferredLocation;
        case UPTKMemAdviseUnsetReadMostly:
            return cudaMemAdviseUnsetReadMostly;
        default:
            ERROR_INVALID_ENUM();
    }
}

cudaMemAllocationType UPTKMemAllocationTypeTocudaMemAllocationType(enum UPTKMemAllocationType para) {
    switch (para) {
        case UPTKMemAllocationTypeInvalid:
            return cudaMemAllocationTypeInvalid;
        case UPTKMemAllocationTypeMax:
            return cudaMemAllocationTypeMax;
        case UPTKMemAllocationTypePinned:
            return cudaMemAllocationTypePinned;
        default:
            ERROR_INVALID_ENUM();
    }
}

UPTKMemAllocationType cudaMemAllocationTypeToUPTKMemAllocationType(enum cudaMemAllocationType para) {
    switch (para)
    {
        case cudaMemAllocationTypeInvalid:
            return UPTKMemAllocationTypeInvalid;
        case cudaMemAllocationTypeMax:
            return UPTKMemAllocationTypeMax;
        case cudaMemAllocationTypePinned:
            return UPTKMemAllocationTypePinned;
        default:
            ERROR_INVALID_ENUM();
    }
}

cudaMemAllocationHandleType UPTKMemAllocationHandleTypeTocudaMemAllocationHandleType(enum UPTKMemAllocationHandleType para) {
    switch (para) {
        case UPTKMemHandleTypeNone:
            return cudaMemHandleTypeNone;
        case UPTKMemHandleTypePosixFileDescriptor:
            return cudaMemHandleTypePosixFileDescriptor;
        case UPTKMemHandleTypeWin32:
            return cudaMemHandleTypeWin32;
        case UPTKMemHandleTypeWin32Kmt:
            return cudaMemHandleTypeWin32Kmt;
        default:
            ERROR_INVALID_ENUM();
    }
}

UPTKMemAllocationHandleType cudaMemAllocationHandleTypeToUPTKMemAllocationHandleType(enum cudaMemAllocationHandleType para) {
    switch (para) {
        case cudaMemHandleTypeNone:
            return UPTKMemHandleTypeNone;
        case cudaMemHandleTypePosixFileDescriptor:
            return UPTKMemHandleTypePosixFileDescriptor;
        case cudaMemHandleTypeWin32:
            return UPTKMemHandleTypeWin32;
        case cudaMemHandleTypeWin32Kmt:
            return UPTKMemHandleTypeWin32Kmt;
        default:
            ERROR_INVALID_ENUM();
    }
}

cudaMemLocationType UPTKMemLocationTypeTocudaMemLocationType(enum UPTKMemLocationType para) {
    switch (para) {
        case UPTKMemLocationTypeDevice:
            return cudaMemLocationTypeDevice;
        case UPTKMemLocationTypeInvalid:
            return cudaMemLocationTypeInvalid;
        case UPTKMemLocationTypeHost:
            return cudaMemLocationTypeHost;
        case UPTKMemLocationTypeHostNuma:   
            return cudaMemLocationTypeHostNuma;
        case UPTKMemLocationTypeHostNumaCurrent:
            return cudaMemLocationTypeHostNumaCurrent;
        default:
            ERROR_INVALID_ENUM();
    }
}

UPTKMemLocationType cudaMemLocationTypeToUPTKMemLocationType(enum cudaMemLocationType para) {
    switch (para) {
        case cudaMemLocationTypeDevice:
            return UPTKMemLocationTypeDevice;
        case cudaMemLocationTypeInvalid:
            return UPTKMemLocationTypeInvalid;
        case cudaMemLocationTypeHost:
            return UPTKMemLocationTypeHost;
        case cudaMemLocationTypeHostNuma:
            return UPTKMemLocationTypeHostNuma;
        case cudaMemLocationTypeHostNumaCurrent:
            return UPTKMemLocationTypeHostNumaCurrent;
        default:
            ERROR_INVALID_ENUM();
    }
}

cudaMemPoolAttr UPTKMemPoolAttrTocudaMemPoolAttr(enum UPTKMemPoolAttr para) {
    switch (para) {
        case UPTKMemPoolAttrUsedMemHigh:
            return cudaMemPoolAttrUsedMemHigh;
        case UPTKMemPoolAttrUsedMemCurrent:
            return cudaMemPoolAttrUsedMemCurrent;
        case UPTKMemPoolAttrReservedMemHigh:
            return cudaMemPoolAttrReservedMemHigh;
        case UPTKMemPoolAttrReservedMemCurrent:
            return cudaMemPoolAttrReservedMemCurrent;
        case UPTKMemPoolAttrReleaseThreshold:
            return cudaMemPoolAttrReleaseThreshold;
        case UPTKMemPoolReuseAllowInternalDependencies:
            return cudaMemPoolReuseAllowInternalDependencies;
        case UPTKMemPoolReuseAllowOpportunistic:
            return cudaMemPoolReuseAllowOpportunistic;
        case UPTKMemPoolReuseFollowEventDependencies:
            return cudaMemPoolReuseFollowEventDependencies;
        default:
            ERROR_INVALID_ENUM();
    }
}

cudaMemRangeAttribute UPTKMemRangeAttributeTocudaMemRangeAttribute(enum UPTKMemRangeAttribute para) {
    switch (para) {
        case UPTKMemRangeAttributeAccessedBy:
            return cudaMemRangeAttributeAccessedBy;
        case UPTKMemRangeAttributeLastPrefetchLocation:
            return cudaMemRangeAttributeLastPrefetchLocation;
        case UPTKMemRangeAttributePreferredLocation:
            return cudaMemRangeAttributePreferredLocation;
        case UPTKMemRangeAttributeReadMostly:
            return cudaMemRangeAttributeReadMostly;
        default:
            ERROR_INVALID_ENUM();
    }
}

enum UPTKMemRangeAttribute cudaMemRangeAttributeToUPTKMemRangeAttribute(cudaMemRangeAttribute para) {
    switch (para) {
        case cudaMemRangeAttributeAccessedBy:
            return UPTKMemRangeAttributeAccessedBy;
        case cudaMemRangeAttributeLastPrefetchLocation:
            return UPTKMemRangeAttributeLastPrefetchLocation;
        case cudaMemRangeAttributePreferredLocation:
            return UPTKMemRangeAttributePreferredLocation;
        case cudaMemRangeAttributeReadMostly:
            return UPTKMemRangeAttributeReadMostly;
        default:
            ERROR_INVALID_ENUM();
    }
}

cudaMemcpyKind UPTKMemcpyKindTocudaMemcpyKind(enum UPTKMemcpyKind para) {
    switch (para) {
        case UPTKMemcpyDefault:
            return cudaMemcpyDefault;
        case UPTKMemcpyDeviceToDevice:
            return cudaMemcpyDeviceToDevice;
        case UPTKMemcpyDeviceToHost:
            return cudaMemcpyDeviceToHost;
        case UPTKMemcpyHostToDevice:
            return cudaMemcpyHostToDevice;
        case UPTKMemcpyHostToHost:
            return cudaMemcpyHostToHost;
        default:
            ERROR_INVALID_ENUM();
    }
}

enum UPTKMemcpyKind cudaMemcpyKindToUPTKMemcpyKind(cudaMemcpyKind para) {
    switch (para) {
        case cudaMemcpyDefault:
            return UPTKMemcpyDefault;
        case cudaMemcpyDeviceToDevice:
            return UPTKMemcpyDeviceToDevice;
        case cudaMemcpyDeviceToHost:
            return UPTKMemcpyDeviceToHost;
        case cudaMemcpyHostToDevice:
            return UPTKMemcpyHostToDevice;
        case cudaMemcpyHostToHost:
            return UPTKMemcpyHostToHost;
        default:
            ERROR_INVALID_ENUM();
    }
}

cudaMemoryType UPTKMemoryTypeTocudaMemoryType(enum UPTKMemoryType para) {
    switch (para) {
        case UPTKMemoryTypeDevice:
            return cudaMemoryTypeDevice;
        case UPTKMemoryTypeHost:
            return cudaMemoryTypeHost;
        case UPTKMemoryTypeManaged:
            return cudaMemoryTypeManaged;
        case UPTKMemoryTypeUnregistered:
            return cudaMemoryTypeUnregistered;
        default:
            ERROR_INVALID_ENUM();
    }
}

enum UPTKMemoryType cudaMemoryTypeToUPTKMemoryType(cudaMemoryType para) {
    switch (para) {
        case cudaMemoryTypeDevice:
            return UPTKMemoryTypeDevice;
        case cudaMemoryTypeHost:
            return UPTKMemoryTypeHost;
        case cudaMemoryTypeManaged:
            return UPTKMemoryTypeManaged;
        case cudaMemoryTypeUnregistered:
            return UPTKMemoryTypeUnregistered;
        default:
            ERROR_INVALID_ENUM();
    }
}

enum cudaTextureReadMode UPTKTextureReadModeTocudaTextureReadMode(enum UPTKTextureReadMode para) {
    switch (para) {
        case UPTKReadModeElementType:
            return cudaReadModeElementType;
        case UPTKReadModeNormalizedFloat:
            return cudaReadModeNormalizedFloat;
        default:
            ERROR_INVALID_ENUM();
    }
}

cudaResourceViewFormat UPTKResourceViewFormatTocudaResourceViewFormat(enum UPTKResourceViewFormat para) {
    switch (para) {
        case UPTKResViewFormatFloat1:
            return cudaResViewFormatFloat1;
        case UPTKResViewFormatFloat2:
            return cudaResViewFormatFloat2;
        case UPTKResViewFormatFloat4:
            return cudaResViewFormatFloat4;
        case UPTKResViewFormatHalf1:
            return cudaResViewFormatHalf1;
        case UPTKResViewFormatHalf2:
            return cudaResViewFormatHalf2;
        case UPTKResViewFormatHalf4:
            return cudaResViewFormatHalf4;
        case UPTKResViewFormatNone:
            return cudaResViewFormatNone;
        case UPTKResViewFormatSignedBlockCompressed4:
            return cudaResViewFormatSignedBlockCompressed4;
        case UPTKResViewFormatSignedBlockCompressed5:
            return cudaResViewFormatSignedBlockCompressed5;
        case UPTKResViewFormatSignedBlockCompressed6H:
            return cudaResViewFormatSignedBlockCompressed6H;
        case UPTKResViewFormatSignedChar1:
            return cudaResViewFormatSignedChar1;
        case UPTKResViewFormatSignedChar2:
            return cudaResViewFormatSignedChar2;
        case UPTKResViewFormatSignedChar4:
            return cudaResViewFormatSignedChar4;
        case UPTKResViewFormatSignedInt1:
            return cudaResViewFormatSignedInt1;
        case UPTKResViewFormatSignedInt2:
            return cudaResViewFormatSignedInt2;
        case UPTKResViewFormatSignedInt4:
            return cudaResViewFormatSignedInt4;
        case UPTKResViewFormatSignedShort1:
            return cudaResViewFormatSignedShort1;
        case UPTKResViewFormatSignedShort2:
            return cudaResViewFormatSignedShort2;
        case UPTKResViewFormatSignedShort4:
            return cudaResViewFormatSignedShort4;
        case UPTKResViewFormatUnsignedBlockCompressed1:
            return cudaResViewFormatUnsignedBlockCompressed1;
        case UPTKResViewFormatUnsignedBlockCompressed2:
            return cudaResViewFormatUnsignedBlockCompressed2;
        case UPTKResViewFormatUnsignedBlockCompressed3:
            return cudaResViewFormatUnsignedBlockCompressed3;
        case UPTKResViewFormatUnsignedBlockCompressed4:
            return cudaResViewFormatUnsignedBlockCompressed4;
        case UPTKResViewFormatUnsignedBlockCompressed5:
            return cudaResViewFormatUnsignedBlockCompressed5;
        case UPTKResViewFormatUnsignedBlockCompressed6H:
            return cudaResViewFormatUnsignedBlockCompressed6H;
        case UPTKResViewFormatUnsignedBlockCompressed7:
            return cudaResViewFormatUnsignedBlockCompressed7;
        case UPTKResViewFormatUnsignedChar1:
            return cudaResViewFormatUnsignedChar1;
        case UPTKResViewFormatUnsignedChar2:
            return cudaResViewFormatUnsignedChar2;
        case UPTKResViewFormatUnsignedChar4:
            return cudaResViewFormatUnsignedChar4;
        case UPTKResViewFormatUnsignedInt1:
            return cudaResViewFormatUnsignedInt1;
        case UPTKResViewFormatUnsignedInt2:
            return cudaResViewFormatUnsignedInt2;
        case UPTKResViewFormatUnsignedInt4:
            return cudaResViewFormatUnsignedInt4;
        case UPTKResViewFormatUnsignedShort1:
            return cudaResViewFormatUnsignedShort1;
        case UPTKResViewFormatUnsignedShort2:
            return cudaResViewFormatUnsignedShort2;
        case UPTKResViewFormatUnsignedShort4:
            return cudaResViewFormatUnsignedShort4;
        default:
            ERROR_INVALID_ENUM();
    }
}

enum UPTKResourceViewFormat cudaResourceViewFormatToUPTKResourceViewFormat(cudaResourceViewFormat para) {
    switch (para) {
        case cudaResViewFormatFloat1:
            return UPTKResViewFormatFloat1;
        case cudaResViewFormatFloat2:
            return UPTKResViewFormatFloat2;
        case cudaResViewFormatFloat4:
            return UPTKResViewFormatFloat4;
        case cudaResViewFormatHalf1:
            return UPTKResViewFormatHalf1;
        case cudaResViewFormatHalf2:
            return UPTKResViewFormatHalf2;
        case cudaResViewFormatHalf4:
            return UPTKResViewFormatHalf4;
        case cudaResViewFormatNone:
            return UPTKResViewFormatNone;
        case cudaResViewFormatSignedBlockCompressed4:
            return UPTKResViewFormatSignedBlockCompressed4;
        case cudaResViewFormatSignedBlockCompressed5:
            return UPTKResViewFormatSignedBlockCompressed5;
        case cudaResViewFormatSignedBlockCompressed6H:
            return UPTKResViewFormatSignedBlockCompressed6H;
        case cudaResViewFormatSignedChar1:
            return UPTKResViewFormatSignedChar1;
        case cudaResViewFormatSignedChar2:
            return UPTKResViewFormatSignedChar2;
        case cudaResViewFormatSignedChar4:
            return UPTKResViewFormatSignedChar4;
        case cudaResViewFormatSignedInt1:
            return UPTKResViewFormatSignedInt1;
        case cudaResViewFormatSignedInt2:
            return UPTKResViewFormatSignedInt2;
        case cudaResViewFormatSignedInt4:
            return UPTKResViewFormatSignedInt4;
        case cudaResViewFormatSignedShort1:
            return UPTKResViewFormatSignedShort1;
        case cudaResViewFormatSignedShort2:
            return UPTKResViewFormatSignedShort2;
        case cudaResViewFormatSignedShort4:
            return UPTKResViewFormatSignedShort4;
        case cudaResViewFormatUnsignedBlockCompressed1:
            return UPTKResViewFormatUnsignedBlockCompressed1;
        case cudaResViewFormatUnsignedBlockCompressed2:
            return UPTKResViewFormatUnsignedBlockCompressed2;
        case cudaResViewFormatUnsignedBlockCompressed3:
            return UPTKResViewFormatUnsignedBlockCompressed3;
        case cudaResViewFormatUnsignedBlockCompressed4:
            return UPTKResViewFormatUnsignedBlockCompressed4;
        case cudaResViewFormatUnsignedBlockCompressed5:
            return UPTKResViewFormatUnsignedBlockCompressed5;
        case cudaResViewFormatUnsignedBlockCompressed6H:
            return UPTKResViewFormatUnsignedBlockCompressed6H;
        case cudaResViewFormatUnsignedBlockCompressed7:
            return UPTKResViewFormatUnsignedBlockCompressed7;
        case cudaResViewFormatUnsignedChar1:
            return UPTKResViewFormatUnsignedChar1;
        case cudaResViewFormatUnsignedChar2:
            return UPTKResViewFormatUnsignedChar2;
        case cudaResViewFormatUnsignedChar4:
            return UPTKResViewFormatUnsignedChar4;
        case cudaResViewFormatUnsignedInt1:
            return UPTKResViewFormatUnsignedInt1;
        case cudaResViewFormatUnsignedInt2:
            return UPTKResViewFormatUnsignedInt2;
        case cudaResViewFormatUnsignedInt4:
            return UPTKResViewFormatUnsignedInt4;
        case cudaResViewFormatUnsignedShort1:
            return UPTKResViewFormatUnsignedShort1;
        case cudaResViewFormatUnsignedShort2:
            return UPTKResViewFormatUnsignedShort2;
        case cudaResViewFormatUnsignedShort4:
            return UPTKResViewFormatUnsignedShort4;
        default:
            ERROR_INVALID_ENUM();
    }
}

cudaResourceType UPTKResourceTypeTocudaResourceType(enum UPTKResourceType para) {
    switch (para) {
        case UPTKResourceTypeArray:
            return cudaResourceTypeArray;
        case UPTKResourceTypeLinear:
            return cudaResourceTypeLinear;
        case UPTKResourceTypeMipmappedArray:
            return cudaResourceTypeMipmappedArray;
        case UPTKResourceTypePitch2D:
            return cudaResourceTypePitch2D;
        default:
            ERROR_INVALID_ENUM();
    }
}

enum UPTKResourceType cudaResourceTypeToUPTKResourceType(cudaResourceType para) {
    switch (para) {
        case cudaResourceTypeArray:
            return UPTKResourceTypeArray;
        case cudaResourceTypeLinear:
            return UPTKResourceTypeLinear;
        case cudaResourceTypeMipmappedArray:
            return UPTKResourceTypeMipmappedArray;
        case cudaResourceTypePitch2D:
            return UPTKResourceTypePitch2D;
        default:
            ERROR_INVALID_ENUM();
    }
}

cudaSharedMemConfig UPTKSharedMemConfigTocudaSharedMemConfig(enum UPTKSharedMemConfig para) {
    switch (para) {
        case UPTKSharedMemBankSizeDefault:
            return cudaSharedMemBankSizeDefault;
        case UPTKSharedMemBankSizeEightByte:
            return cudaSharedMemBankSizeEightByte;
        case UPTKSharedMemBankSizeFourByte:
            return cudaSharedMemBankSizeFourByte;
        default:
            ERROR_INVALID_ENUM();
    }
}

enum UPTKSharedMemConfig cudaSharedMemConfigToUPTKSharedMemConfig(cudaSharedMemConfig para) {
    switch (para) {
        case cudaSharedMemBankSizeDefault:
            return UPTKSharedMemBankSizeDefault;
        case cudaSharedMemBankSizeEightByte:
            return UPTKSharedMemBankSizeEightByte;
        case cudaSharedMemBankSizeFourByte:
            return UPTKSharedMemBankSizeFourByte;
        default:
            ERROR_INVALID_ENUM();
    }
}

// enum cudaSharedCarveout UPTKSharedCarveoutTocudaSharedCarveout(enum UPTKSharedCarveout para) {
//     switch (para) {
//         case UPTKSharedmemCarveoutDefault:
//             return cudaSharedmemCarveoutDefault;
//         case UPTKSharedmemCarveoutMaxL1:
//             return cudaSharedmemCarveoutMaxL1;
//         case UPTKSharedmemCarveoutMaxShared:
//             return cudaSharedmemCarveoutMaxShared;
//         default:
//             ERROR_INVALID_ENUM();
//     }
// }

// enum cudaStreamAttrID UPTKStreamAttrIDTocudaStreamAttrID(enum UPTKStreamAttrID para) {
//     switch (para) {
//         case UPTKLaunchAttributeIgnore:
//             return cudaLaunchAttributeIgnore;
//         case UPTKLaunchAttributeAccessPolicyWindow:
//             return cudaLaunchAttributeAccessPolicyWindow;
//         case UPTKLaunchAttributeCooperative:
//             return cudaLaunchAttributeCooperative;
//         case UPTKLaunchAttributeSynchronizationPolicy:
//             return cudaLaunchAttributeSynchronizationPolicy;
//         case UPTKLaunchAttributeClusterDimension:
//             return cudaLaunchAttributeClusterDimension;
//         case UPTKLaunchAttributeClusterSchedulingPolicyPreference:
//             return cudaLaunchAttributeClusterSchedulingPolicyPreference;
//         case UPTKLaunchAttributeProgrammaticStreamSerialization:
//             return cudaLaunchAttributeProgrammaticStreamSerialization;
//         case UPTKLaunchAttributeProgrammaticEvent:
//             return cudaLaunchAttributeProgrammaticEvent;
//         case UPTKLaunchAttributePriority:
//             return cudaLaunchAttributePriority;
//         default:
//             ERROR_INVALID_ENUM();
//     }
// }

cudaStreamCaptureMode UPTKStreamCaptureModeTocudaStreamCaptureMode(enum UPTKStreamCaptureMode para) {
    switch (para) {
        case UPTKStreamCaptureModeGlobal:
            return cudaStreamCaptureModeGlobal;
        case UPTKStreamCaptureModeRelaxed:
            return cudaStreamCaptureModeRelaxed;
        case UPTKStreamCaptureModeThreadLocal:
            return cudaStreamCaptureModeThreadLocal;
        default:
            ERROR_INVALID_ENUM();
    }
}

enum UPTKStreamCaptureMode cudaStreamCaptureModeToUPTKStreamCaptureMode(cudaStreamCaptureMode para) {
    switch (para) {
        case cudaStreamCaptureModeGlobal:
            return UPTKStreamCaptureModeGlobal;
        case cudaStreamCaptureModeRelaxed:
            return UPTKStreamCaptureModeRelaxed;
        case cudaStreamCaptureModeThreadLocal:
            return UPTKStreamCaptureModeThreadLocal;
        default:
            ERROR_INVALID_ENUM();
    }
}

cudaStreamCaptureStatus UPTKStreamCaptureStatusTocudaStreamCaptureStatus(enum UPTKStreamCaptureStatus para) {
    switch (para) {
        case UPTKStreamCaptureStatusActive:
            return cudaStreamCaptureStatusActive;
        case UPTKStreamCaptureStatusInvalidated:
            return cudaStreamCaptureStatusInvalidated;
        case UPTKStreamCaptureStatusNone:
            return cudaStreamCaptureStatusNone;
        default:
            ERROR_INVALID_ENUM();
    }
}

/*enum UPTKStreamCaptureStatus cudaStreamCaptureStatusToUPTKStreamCaptureStatus(cudaStreamCaptureStatus para) {
    switch (para) {
        case cudaStreamCaptureStatusActive:
            return UPTKStreamCaptureStatusActive;
        case cudaStreamCaptureStatusInvalidated:
            return UPTKStreamCaptureStatusInvalidated;
        case cudaStreamCaptureStatusNone:
            return UPTKStreamCaptureStatusNone;
        default:
            ERROR_INVALID_ENUM();
    }
}*/

enum cudaSynchronizationPolicy UPTKSynchronizationPolicyTocudaSynchronizationPolicy(enum UPTKSynchronizationPolicy para) {
    switch (para) {
        case UPTKSyncPolicyAuto:
            return cudaSyncPolicyAuto;
        case UPTKSyncPolicyBlockingSync:
            return cudaSyncPolicyBlockingSync;
        case UPTKSyncPolicySpin:
            return cudaSyncPolicySpin;
        case UPTKSyncPolicyYield:
            return cudaSyncPolicyYield;
        default:
            ERROR_INVALID_ENUM();
    }
}

/*void UPTKDevicePropTocudaDeviceProp(const struct UPTKDeviceProp * UPTK_para, cudaDeviceProp * cuda_para) {
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    strncpy(cuda_para->name, UPTK_para->name, 256);
    cuda_para->totalGlobalMem = UPTK_para->totalGlobalMem;
    cuda_para->sharedMemPerBlock = UPTK_para->sharedMemPerBlock;
    cuda_para->regsPerBlock = UPTK_para->regsPerBlock;
    cuda_para->warpSize = UPTK_para->warpSize;
    cuda_para->maxThreadsPerBlock = UPTK_para->maxThreadsPerBlock;
    for (int i = 0; i < 3; i++) {
        cuda_para->maxThreadsDim[i] = UPTK_para->maxThreadsDim[i];
        cuda_para->maxGridSize[i] = UPTK_para->maxGridSize[i];
    }
    cuda_para->clockRate = UPTK_para->clockRate;
    cuda_para->memoryClockRate = UPTK_para->memoryClockRate;
    cuda_para->memoryBusWidth = UPTK_para->memoryBusWidth;
    cuda_para->totalConstMem = UPTK_para->totalConstMem;
    cuda_para->major = UPTK_para->major;
    cuda_para->minor = UPTK_para->minor;
    cuda_para->multiProcessorCount = UPTK_para->multiProcessorCount;
    cuda_para->l2CacheSize = UPTK_para->l2CacheSize;
    cuda_para->maxThreadsPerMultiProcessor = UPTK_para->maxThreadsPerMultiProcessor;
    cuda_para->computeMode = UPTK_para->computeMode;
    cuda_para->clockInstructionRate = UPTK_para->clockRate; // Same as clock-rate:

    int ccVers = cuda_para->major * 100 + cuda_para->minor * 10;
    cuda_para->arch.hasGlobalInt32Atomics = (ccVers >= 110);
    cuda_para->arch.hasGlobalFloatAtomicExch = (ccVers >= 110);
    cuda_para->arch.hasSharedInt32Atomics = (ccVers >= 120);
    cuda_para->arch.hasSharedFloatAtomicExch = (ccVers >= 120);
    cuda_para->arch.hasFloatAtomicAdd = (ccVers >= 200);
    cuda_para->arch.hasGlobalInt64Atomics = (ccVers >= 120);
    cuda_para->arch.hasSharedInt64Atomics = (ccVers >= 110);
    cuda_para->arch.hasDoubles = (ccVers >= 130);
    cuda_para->arch.hasWarpVote = (ccVers >= 120);
    cuda_para->arch.hasWarpBallot = (ccVers >= 200);
    cuda_para->arch.hasWarpShuffle = (ccVers >= 300);
    cuda_para->arch.hasFunnelShift = (ccVers >= 350);
    cuda_para->arch.hasThreadFenceSystem = (ccVers >= 200);
    cuda_para->arch.hasSyncThreadsExt = (ccVers >= 200);
    cuda_para->arch.hasSurfaceFuncs = (ccVers >= 200);
    cuda_para->arch.has3dGrid = (ccVers >= 200);
    cuda_para->arch.hasDynamicParallelism = (ccVers >= 350);

    cuda_para->concurrentKernels = UPTK_para->concurrentKernels;
    cuda_para->pciDomainID = UPTK_para->pciDomainID;
    cuda_para->pciBusID = UPTK_para->pciBusID;
    cuda_para->pciDeviceID = UPTK_para->pciDeviceID;
    cuda_para->maxSharedMemoryPerMultiProcessor = UPTK_para->sharedMemPerMultiprocessor;
    cuda_para->isMultiGpuBoard = UPTK_para->isMultiGpuBoard;
    cuda_para->canMapHostMemory = UPTK_para->canMapHostMemory;
    cuda_para->gcnArch = 0; // Not a GCN arch
    cuda_para->integrated = UPTK_para->integrated;
    cuda_para->cooperativeLaunch = UPTK_para->cooperativeLaunch;
    cuda_para->cooperativeMultiDeviceLaunch = UPTK_para->cooperativeMultiDeviceLaunch;
    cuda_para->cooperativeMultiDeviceUnmatchedFunc = 0;
    cuda_para->cooperativeMultiDeviceUnmatchedGridDim = 0;
    cuda_para->cooperativeMultiDeviceUnmatchedBlockDim = 0;
    cuda_para->cooperativeMultiDeviceUnmatchedSharedMem = 0;

    cuda_para->maxTexture1D    = UPTK_para->maxTexture1D;
    cuda_para->maxTexture2D[0] = UPTK_para->maxTexture2D[0];
    cuda_para->maxTexture2D[1] = UPTK_para->maxTexture2D[1];
    cuda_para->maxTexture3D[0] = UPTK_para->maxTexture3D[0];
    cuda_para->maxTexture3D[1] = UPTK_para->maxTexture3D[1];
    cuda_para->maxTexture3D[2] = UPTK_para->maxTexture3D[2];

    cuda_para->memPitch                 = UPTK_para->memPitch;
    cuda_para->textureAlignment         = UPTK_para->textureAlignment;
    cuda_para->texturePitchAlignment    = UPTK_para->texturePitchAlignment;
    cuda_para->kernelExecTimeoutEnabled = UPTK_para->kernelExecTimeoutEnabled;
    cuda_para->ECCEnabled               = UPTK_para->ECCEnabled;
    cuda_para->tccDriver                = UPTK_para->tccDriver;
}

void cudaDevicePropToUPTKDeviceProp(const cudaDeviceProp * cuda_para, struct UPTKDeviceProp * UPTK_para) {
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    // Add the gcnArchName of cuda to the device name
    strncpy(UPTK_para->name, cuda_para->name, sizeof(UPTK_para->name) - 1);
    UPTK_para->name[sizeof(UPTK_para->name) - 1] = '\0'; 
    if (strlen(UPTK_para->name) + strlen(cuda_para->gcnArchName) < sizeof(UPTK_para->name) - 1){
        strncat(UPTK_para->name, " ", sizeof(UPTK_para->name) - strlen(UPTK_para->name) - 1); 
        strncat(UPTK_para->name, cuda_para->gcnArchName, sizeof(UPTK_para->name) - strlen(UPTK_para->name) - 1); 
    }
    else {
        fprintf(stderr, "Error: UPTK_para->name array is not large enough to hold the result.\n");
    }
    UPTK_para->totalGlobalMem = cuda_para->totalGlobalMem;
    UPTK_para->sharedMemPerBlock = cuda_para->sharedMemPerBlock;
    UPTK_para->regsPerBlock = cuda_para->regsPerBlock;
    UPTK_para->warpSize = cuda_para->warpSize;
    UPTK_para->maxThreadsPerBlock = cuda_para->maxThreadsPerBlock;
    for (int i = 0; i < 3; i++) {
        UPTK_para->maxThreadsDim[i] = cuda_para->maxThreadsDim[i];
        UPTK_para->maxGridSize[i] = cuda_para->maxGridSize[i];
    }
    UPTK_para->clockRate = cuda_para->clockRate;
    UPTK_para->memoryClockRate = cuda_para->memoryClockRate;
    UPTK_para->memoryBusWidth = cuda_para->memoryBusWidth;
    UPTK_para->totalConstMem = cuda_para->totalConstMem;
    UPTK_para->major = SM_VERSION_MAJOR;   // not equal to cuda
    UPTK_para->minor = SM_VERSION_MINOR;   // not equal to cuda
    UPTK_para->multiProcessorCount = cuda_para->multiProcessorCount;
    UPTK_para->l2CacheSize = cuda_para->l2CacheSize;
    UPTK_para->maxThreadsPerMultiProcessor = cuda_para->maxThreadsPerMultiProcessor;
    UPTK_para->computeMode = cuda_para->computeMode;

    UPTK_para->concurrentKernels = cuda_para->concurrentKernels;
    UPTK_para->pciDomainID = cuda_para->pciDomainID;
    UPTK_para->pciBusID = cuda_para->pciBusID;
    UPTK_para->pciDeviceID = cuda_para->pciDeviceID;
    UPTK_para->sharedMemPerMultiprocessor = cuda_para->maxSharedMemoryPerMultiProcessor;
    UPTK_para->isMultiGpuBoard = cuda_para->isMultiGpuBoard;
    UPTK_para->canMapHostMemory = cuda_para->canMapHostMemory;
    UPTK_para->integrated = cuda_para->integrated;
    UPTK_para->cooperativeLaunch = cuda_para->cooperativeLaunch;
    UPTK_para->cooperativeMultiDeviceLaunch = cuda_para->cooperativeMultiDeviceLaunch;

    UPTK_para->maxTexture1D    = cuda_para->maxTexture1D;
    UPTK_para->maxTexture2D[0] = cuda_para->maxTexture2D[0];
    UPTK_para->maxTexture2D[1] = cuda_para->maxTexture2D[1];
    UPTK_para->maxTexture3D[0] = cuda_para->maxTexture3D[0];
    UPTK_para->maxTexture3D[1] = cuda_para->maxTexture3D[1];
    UPTK_para->maxTexture3D[2] = cuda_para->maxTexture3D[2];

    UPTK_para->memPitch                 = cuda_para->memPitch;
    UPTK_para->textureAlignment         = cuda_para->textureAlignment;
    UPTK_para->texturePitchAlignment    = cuda_para->texturePitchAlignment;
    UPTK_para->kernelExecTimeoutEnabled = cuda_para->kernelExecTimeoutEnabled;
    UPTK_para->ECCEnabled               = cuda_para->ECCEnabled;
    UPTK_para->tccDriver                = cuda_para->tccDriver;

    // Use logical alignment of missing fields from pytorch
    UPTK_para->regsPerMultiprocessor = cuda_para->regsPerBlock;
    UPTK_para->maxBlocksPerMultiProcessor = 64;
}

void cudaDeviceProp_v2ToUPTKDeviceProp(const cudaDeviceProp * cuda_para, struct UPTKDeviceProp * UPTK_para) {
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    // Add the gcnArchName of cuda to the device name
    strncpy(UPTK_para->name, cuda_para->name, sizeof(UPTK_para->name) - 1);
    UPTK_para->name[sizeof(UPTK_para->name) - 1] = '\0'; 
    if (strlen(UPTK_para->name) + strlen(cuda_para->gcnArchName) < sizeof(UPTK_para->name) - 1){
        strncat(UPTK_para->name, " ", sizeof(UPTK_para->name) - strlen(UPTK_para->name) - 1); 
        strncat(UPTK_para->name, cuda_para->gcnArchName, sizeof(UPTK_para->name) - strlen(UPTK_para->name) - 1); 
    }
    else {
        fprintf(stderr, "Error: UPTK_para->name array is not large enough to hold the result.\n");
    }
    UPTK_para->totalGlobalMem = cuda_para->totalGlobalMem;
    UPTK_para->sharedMemPerBlock = cuda_para->sharedMemPerBlock;
    UPTK_para->regsPerBlock = cuda_para->regsPerBlock;
    UPTK_para->warpSize = cuda_para->warpSize;
    UPTK_para->maxThreadsPerBlock = cuda_para->maxThreadsPerBlock;
    for (int i = 0; i < 3; i++) {
        UPTK_para->maxThreadsDim[i] = cuda_para->maxThreadsDim[i];
        UPTK_para->maxGridSize[i] = cuda_para->maxGridSize[i];
    }
    UPTK_para->clockRate = cuda_para->clockRate;
    UPTK_para->memoryClockRate = cuda_para->memoryClockRate;
    UPTK_para->memoryBusWidth = cuda_para->memoryBusWidth;
    UPTK_para->totalConstMem = cuda_para->totalConstMem;
    UPTK_para->major = SM_VERSION_MAJOR;   // not equal to cuda
    UPTK_para->minor = SM_VERSION_MINOR;   // not equal to cuda
    UPTK_para->multiProcessorCount = cuda_para->multiProcessorCount;
    UPTK_para->l2CacheSize = cuda_para->l2CacheSize;
    UPTK_para->maxThreadsPerMultiProcessor = cuda_para->maxThreadsPerMultiProcessor;
    UPTK_para->computeMode = cuda_para->computeMode;

    UPTK_para->concurrentKernels = cuda_para->concurrentKernels;
    UPTK_para->pciDomainID = cuda_para->pciDomainID;
    UPTK_para->pciBusID = cuda_para->pciBusID;
    UPTK_para->pciDeviceID = cuda_para->pciDeviceID;
    
    // Only sharedMemPerMultiprocessor UPTKDeviceProp members
    // Two members of the cudaDeviceProp_t_v2 are sharedMemPerMultiprocessor, maxSharedMemoryPerMultiProcessor
    // In the verification to the environment in P100 NV, UPTK sharedMemPerMultiprocessor maxSharedMemoryPerMultiProcessor should correspond to cuda
    UPTK_para->sharedMemPerMultiprocessor = cuda_para->maxSharedMemoryPerMultiProcessor;

    UPTK_para->isMultiGpuBoard = cuda_para->isMultiGpuBoard;
    UPTK_para->canMapHostMemory = cuda_para->canMapHostMemory;
    UPTK_para->integrated = cuda_para->integrated;
    UPTK_para->cooperativeLaunch = cuda_para->cooperativeLaunch;
    UPTK_para->cooperativeMultiDeviceLaunch = cuda_para->cooperativeMultiDeviceLaunch;

    UPTK_para->maxTexture1D    = cuda_para->maxTexture1D;
    UPTK_para->maxTexture2D[0] = cuda_para->maxTexture2D[0];
    UPTK_para->maxTexture2D[1] = cuda_para->maxTexture2D[1];
    UPTK_para->maxTexture3D[0] = cuda_para->maxTexture3D[0];
    UPTK_para->maxTexture3D[1] = cuda_para->maxTexture3D[1];
    UPTK_para->maxTexture3D[2] = cuda_para->maxTexture3D[2];

    UPTK_para->memPitch                 = cuda_para->memPitch;
    UPTK_para->textureAlignment         = cuda_para->textureAlignment;
    UPTK_para->texturePitchAlignment    = cuda_para->texturePitchAlignment;
    UPTK_para->kernelExecTimeoutEnabled = cuda_para->kernelExecTimeoutEnabled;
    UPTK_para->ECCEnabled               = cuda_para->ECCEnabled;
    UPTK_para->tccDriver                = cuda_para->tccDriver;

    UPTK_para->regsPerMultiprocessor = cuda_para->regsPerMultiprocessor;
    UPTK_para->maxBlocksPerMultiProcessor = cuda_para->maxBlocksPerMultiProcessor;

    memcpy(&(UPTK_para->uuid), &(cuda_para->uuid), sizeof(UPTKUUID_t));
    for(int i = 0; i < 8; i++){
        UPTK_para->luid[i] = cuda_para->luid[i];
    }
    UPTK_para->luidDeviceNodeMask = cuda_para->luidDeviceNodeMask;
    UPTK_para->deviceOverlap = cuda_para->deviceOverlap;
    UPTK_para->maxTexture1DMipmap = cuda_para->maxTexture1DMipmap;
    UPTK_para->maxTexture1DLinear = cuda_para->maxTexture1DLinear;
    
    for(int i = 0; i < 2; i++){
    UPTK_para->maxTexture2DMipmap[i] = cuda_para->maxTexture2DMipmap[i];
    UPTK_para->maxTexture2DGather[i] = cuda_para->maxTexture2DGather[i];
    UPTK_para->maxTexture1DLayered[i] = cuda_para->maxTexture1DLayered[i];
    UPTK_para->maxTextureCubemapLayered[i] = cuda_para->maxTextureCubemapLayered[i];
    UPTK_para->maxSurface2D[i] = cuda_para->maxSurface2D[i];
    UPTK_para->maxSurface1DLayered[i] = cuda_para->maxSurface1DLayered[i];
    UPTK_para->maxSurfaceCubemapLayered[i] = cuda_para->maxSurfaceCubemapLayered[i];
        
    }

    UPTK_para->maxSurface1D = cuda_para->maxSurface1D;

    for(int i = 0; i < 3; i++){
    UPTK_para->maxTexture2DLinear[i] = cuda_para->maxTexture2DLinear[i];
    UPTK_para->maxTexture3DAlt[i] = cuda_para->maxTexture3DAlt[i];
    UPTK_para->maxTextureCubemap = cuda_para->maxTextureCubemap;
    UPTK_para->maxTexture2DLayered[i] = cuda_para->maxTexture2DLayered[i];
    UPTK_para->maxSurface3D[i] = cuda_para->maxSurface3D[i];
    UPTK_para->maxSurface2DLayered[i] = cuda_para->maxSurface2DLayered[i];
    }

    UPTK_para->maxSurfaceCubemap = cuda_para->maxSurfaceCubemap;
    UPTK_para->surfaceAlignment = cuda_para->surfaceAlignment;
    UPTK_para->asyncEngineCount = cuda_para->asyncEngineCount;
    // UPTK_para->unifiedAddressing = cuda_para->unifiedAddressing;
    //  UPTK sample: simpleIPC determines whether the device supports this attribute in the code. 
    //  The underlying hardware does not support unifiedAddressing, but setting unifiedAddressing to 1 does not affect the sample
    UPTK_para->unifiedAddressing = 1;
    UPTK_para->persistingL2CacheMaxSize = cuda_para->persistingL2CacheMaxSize;
    UPTK_para->streamPrioritiesSupported = cuda_para->streamPrioritiesSupported;
    UPTK_para->globalL1CacheSupported = cuda_para->globalL1CacheSupported;
    UPTK_para->localL1CacheSupported = cuda_para->localL1CacheSupported;
    UPTK_para->managedMemory = cuda_para->managedMemory;
    UPTK_para->multiGpuBoardGroupID = cuda_para->multiGpuBoardGroupID;
    UPTK_para->hostNativeAtomicSupported = cuda_para->hostNativeAtomicSupported;
    UPTK_para->singleToDoublePrecisionPerfRatio = cuda_para->singleToDoublePrecisionPerfRatio;
    UPTK_para->pageableMemoryAccess = cuda_para->pageableMemoryAccess;
    UPTK_para->concurrentManagedAccess = cuda_para->concurrentManagedAccess;
    UPTK_para->computePreemptionSupported = cuda_para->computePreemptionSupported;
    UPTK_para->canUseHostPointerForRegisteredMem = cuda_para->canUseHostPointerForRegisteredMem;
    // UPTK_para->sharedMemPerBlockOptin = cuda_para->sharedMemPerBlockOptin;
    // cuda sharedMemPerBlockOptin member returns 0, the actual function is not supported. Replace with the close attribute sharedMemPerBlock.
    UPTK_para->sharedMemPerBlockOptin = cuda_para->sharedMemPerBlock;
    UPTK_para->pageableMemoryAccessUsesHostPageTables = cuda_para->pageableMemoryAccessUsesHostPageTables;
    UPTK_para->directManagedMemAccessFromHost = cuda_para->directManagedMemAccessFromHost;
    UPTK_para->accessPolicyMaxWindowSize = cuda_para->accessPolicyMaxWindowSize;
    UPTK_para->reservedSharedMemPerBlock = cuda_para->reservedSharedMemPerBlock;
    // UPTK 12.6.2:added some members to the UPTKDeviceProp structure.
    UPTK_para->hostRegisterSupported = cuda_para->hostRegisterSupported;
    UPTK_para->sparseUPTKArraySupported = cuda_para->sparsecudaArraySupported;
    UPTK_para->hostRegisterReadOnlySupported = cuda_para->hostRegisterReadOnlySupported;
    UPTK_para->timelineSemaphoreInteropSupported = cuda_para->timelineSemaphoreInteropSupported;
    UPTK_para->memoryPoolsSupported = cuda_para->memoryPoolsSupported;
    UPTK_para->gpuDirectRDMASupported = cuda_para->gpuDirectRDMASupported;
    UPTK_para->gpuDirectRDMAFlushWritesOptions = cuda_para->gpuDirectRDMAFlushWritesOptions;
    UPTK_para->gpuDirectRDMAWritesOrdering = cuda_para->gpuDirectRDMAWritesOrdering;
    UPTK_para->memoryPoolSupportedHandleTypes = cuda_para->memoryPoolSupportedHandleTypes;
    UPTK_para->deferredMappingUPTKArraySupported = cuda_para->deferredMappingcudaArraySupported;
    UPTK_para->ipcEventSupported = cuda_para->ipcEventSupported;
    UPTK_para->clusterLaunch = cuda_para->clusterLaunch;
    UPTK_para->unifiedFunctionPointers = cuda_para->unifiedFunctionPointers;
    memcpy(UPTK_para->reserved2, cuda_para->reserved, 2 * sizeof(int));       
    memcpy(UPTK_para->reserved1, cuda_para->reserved + 2, 1 * sizeof(int));           
    memcpy(UPTK_para->reserved, cuda_para->reserved + 3, 60 * sizeof(int));         
}*/

void UPTKTextureDescTocudaTextureDesc(const struct UPTKTextureDesc * UPTK_para, cudaTextureDesc * cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    for (int i = 0; i < 3; ++i) {
        cuda_para->addressMode[i] = UPTKTextureAddressModeTocudaTextureAddressMode(UPTK_para->addressMode[i]);
    }
    for (int i = 0; i < 4; ++i) {
        cuda_para->borderColor[i] = UPTK_para->borderColor[i];
    }
    cuda_para->filterMode = UPTKTextureFilterModeTocudaTextureFilterMode(UPTK_para->filterMode);
    cuda_para->maxAnisotropy = UPTK_para->maxAnisotropy;
    cuda_para->maxMipmapLevelClamp = UPTK_para->maxMipmapLevelClamp;
    cuda_para->minMipmapLevelClamp = UPTK_para->minMipmapLevelClamp;
    cuda_para->mipmapFilterMode = UPTKTextureFilterModeTocudaTextureFilterMode(UPTK_para->mipmapFilterMode);
    cuda_para->mipmapLevelBias = UPTK_para->mipmapLevelBias;
    cuda_para->normalizedCoords = UPTK_para->normalizedCoords;
    cuda_para->readMode = UPTKTextureReadModeTocudaTextureReadMode(UPTK_para->readMode);
    cuda_para->sRGB = UPTK_para->sRGB;
}

void UPTKLaunchParamsTocudaLaunchParams(const struct UPTKLaunchParams * UPTK_para, cudaLaunchParams * cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->args = UPTK_para->args;
    cuda_para->blockDim = UPTK_para->blockDim;
    cuda_para->func = UPTK_para->func;
    cuda_para->gridDim = UPTK_para->gridDim;
    cuda_para->sharedMem = UPTK_para->sharedMem;
    cuda_para->stream = (cudaStream_t)UPTK_para->stream;
}

void UPTKAccessPolicyWindowTocudaAccessPolicyWindow(const struct UPTKAccessPolicyWindow * UPTK_para, cudaAccessPolicyWindow * cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->base_ptr = UPTK_para->base_ptr;
    cuda_para->hitProp = UPTKAccessPropertyTocudaAccessProperty(UPTK_para->hitProp);
    cuda_para->hitRatio = UPTK_para->hitRatio;
    cuda_para->missProp = UPTKAccessPropertyTocudaAccessProperty(UPTK_para->missProp);
    cuda_para->num_bytes = UPTK_para->num_bytes;
}

void cudaAccessPolicyWindowToUPTKAccessPolicyWindow(const cudaAccessPolicyWindow * cuda_para, struct UPTKAccessPolicyWindow * UPTK_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    UPTK_para->base_ptr = cuda_para->base_ptr;
    UPTK_para->hitProp = cudaAccessPropertyToUPTKAccessProperty(cuda_para->hitProp);
    UPTK_para->hitRatio = cuda_para->hitRatio;
    UPTK_para->missProp = cudaAccessPropertyToUPTKAccessProperty(cuda_para->missProp);
    UPTK_para->num_bytes = cuda_para->num_bytes;
}

void UPTKFuncAttributesTocudaFuncAttributes(const struct UPTKFuncAttributes * UPTK_para, cudaFuncAttributes * cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->binaryVersion = UPTK_para->binaryVersion;
    cuda_para->cacheModeCA = UPTK_para->cacheModeCA;
    cuda_para->constSizeBytes = UPTK_para->constSizeBytes;
    cuda_para->localSizeBytes = UPTK_para->localSizeBytes;
    cuda_para->maxDynamicSharedSizeBytes = UPTK_para->maxDynamicSharedSizeBytes;
    cuda_para->maxThreadsPerBlock = UPTK_para->maxThreadsPerBlock;
    cuda_para->numRegs = UPTK_para->numRegs;
    cuda_para->preferredShmemCarveout = UPTK_para->preferredShmemCarveout;
    cuda_para->ptxVersion = UPTK_para->ptxVersion;
    cuda_para->sharedSizeBytes = UPTK_para->sharedSizeBytes;
}

void cudaFuncAttributesToUPTKFuncAttributes(const cudaFuncAttributes * cuda_para, struct UPTKFuncAttributes * UPTK_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    UPTK_para->binaryVersion = cuda_para->binaryVersion;
    UPTK_para->cacheModeCA = cuda_para->cacheModeCA;
    UPTK_para->constSizeBytes = cuda_para->constSizeBytes;
    UPTK_para->localSizeBytes = cuda_para->localSizeBytes;
    UPTK_para->maxDynamicSharedSizeBytes = cuda_para->maxDynamicSharedSizeBytes;
    UPTK_para->maxThreadsPerBlock = cuda_para->maxThreadsPerBlock;
    UPTK_para->numRegs = cuda_para->numRegs;
    UPTK_para->preferredShmemCarveout = cuda_para->preferredShmemCarveout;
    UPTK_para->ptxVersion = cuda_para->ptxVersion;
    UPTK_para->sharedSizeBytes = cuda_para->sharedSizeBytes;
}

void UPTKKernelNodeParamsTocudaKernelNodeParams(const struct UPTKKernelNodeParams * UPTK_para, cudaKernelNodeParams * cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->blockDim = UPTK_para->blockDim;
    cuda_para->extra = UPTK_para->extra;
    cuda_para->func = UPTK_para->func;
    cuda_para->gridDim = UPTK_para->gridDim;
    cuda_para->kernelParams = UPTK_para->kernelParams;
    cuda_para->sharedMemBytes = UPTK_para->sharedMemBytes;
}

void cudaKernelNodeParamsToUPTKKernelNodeParams(const cudaKernelNodeParams * cuda_para, struct UPTKKernelNodeParams * UPTK_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    UPTK_para->blockDim = cuda_para->blockDim;
    UPTK_para->extra = cuda_para->extra;
    UPTK_para->func = cuda_para->func;
    UPTK_para->gridDim = cuda_para->gridDim;
    UPTK_para->kernelParams = cuda_para->kernelParams;
    UPTK_para->sharedMemBytes = cuda_para->sharedMemBytes;
}

void UPTKResourceViewDescTocudaResourceViewDesc(const struct UPTKResourceViewDesc * UPTK_para, struct cudaResourceViewDesc * cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->depth = UPTK_para->depth;
    cuda_para->firstLayer = UPTK_para->firstLayer;
    cuda_para->firstMipmapLevel = UPTK_para->firstMipmapLevel;
    cuda_para->format = UPTKResourceViewFormatTocudaResourceViewFormat(UPTK_para->format);
    cuda_para->height = UPTK_para->height;
    cuda_para->lastLayer = UPTK_para->lastLayer;
    cuda_para->lastMipmapLevel = UPTK_para->lastMipmapLevel;
    cuda_para->width = UPTK_para->width;
}

void UPTKExtentTocudaExtent(const struct UPTKExtent * UPTK_para, cudaExtent * cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->depth = UPTK_para->depth;
    cuda_para->height = UPTK_para->height;
    cuda_para->width = UPTK_para->width;
}

void cudaExtentToUPTKExtent(const cudaExtent * cuda_para, struct UPTKExtent * UPTK_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    UPTK_para->depth = cuda_para->depth;
    UPTK_para->height = cuda_para->height;
    UPTK_para->width = cuda_para->width;
}

void cudaPointerAttributesToUPTKPointerAttributes(const cudaPointerAttributes * cuda_para, struct UPTKPointerAttributes * UPTK_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    UPTK_para->device = cuda_para->device;
    UPTK_para->devicePointer = cuda_para->devicePointer;
    UPTK_para->hostPointer = cuda_para->hostPointer;
    UPTK_para->type = cudaMemoryTypeToUPTKMemoryType(cuda_para->type);
}

void UPTKMemsetParamsTocudaMemsetParams(const struct UPTKMemsetParams * UPTK_para, cudaMemsetParams * cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->dst = UPTK_para->dst;
    cuda_para->elementSize = UPTK_para->elementSize;
    cuda_para->height = UPTK_para->height;
    cuda_para->pitch = UPTK_para->pitch;
    cuda_para->value = UPTK_para->value;
    cuda_para->width = UPTK_para->width;
}

void cudaMemsetParamsToUPTKMemsetParams(const cudaMemsetParams * cuda_para, struct UPTKMemsetParams * UPTK_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    UPTK_para->dst = cuda_para->dst;
    UPTK_para->elementSize = cuda_para->elementSize;
    UPTK_para->height = cuda_para->height;
    UPTK_para->pitch = cuda_para->pitch;
    UPTK_para->value = cuda_para->value;
    UPTK_para->width = cuda_para->width;
}

void UPTKMemcpy3DParmsTocudaMemcpy3DParms(const struct UPTKMemcpy3DParms * UPTK_para, cudaMemcpy3DParms * cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->dstArray = (cudaArray_t)UPTK_para->dstArray;
    UPTKPosTocudaPos(&(UPTK_para->dstPos), &(cuda_para->dstPos));
    UPTKPitchedPtrTocudaPitchedPtr(&(UPTK_para->dstPtr), &(cuda_para->dstPtr));
    UPTKExtentTocudaExtent(&(UPTK_para->extent), &(cuda_para->extent));
    cuda_para->kind = UPTKMemcpyKindTocudaMemcpyKind(UPTK_para->kind);
    cuda_para->srcArray = (cudaArray_t)UPTK_para->srcArray;
    UPTKPosTocudaPos(&(UPTK_para->srcPos), &(cuda_para->srcPos));
    UPTKPitchedPtrTocudaPitchedPtr(&(UPTK_para->srcPtr), &(cuda_para->srcPtr));
}

void cudaMemcpy3DParmsToUPTKMemcpy3DParms(const cudaMemcpy3DParms * cuda_para, struct UPTKMemcpy3DParms * UPTK_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    UPTK_para->dstArray = (UPTKArray_t)cuda_para->dstArray;
    cudaPosToUPTKPos(&(cuda_para->dstPos), &(UPTK_para->dstPos));
    cudaPitchedPtrToUPTKPitchedPtr(&(cuda_para->dstPtr), &(UPTK_para->dstPtr));
    cudaExtentToUPTKExtent(&(cuda_para->extent), &(UPTK_para->extent));
    UPTK_para->kind = cudaMemcpyKindToUPTKMemcpyKind(cuda_para->kind);
    UPTK_para->srcArray = (UPTKArray_t)cuda_para->srcArray;
    cudaPosToUPTKPos(&(cuda_para->srcPos), &(UPTK_para->srcPos));
    cudaPitchedPtrToUPTKPitchedPtr(&(cuda_para->srcPtr), &(UPTK_para->srcPtr));
}

void UPTKChannelFormatDescTocudaChannelFormatDesc(const struct UPTKChannelFormatDesc * UPTK_para, cudaChannelFormatDesc * cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->f = UPTKChannelFormatKindTocudaChannelFormatKind(UPTK_para->f);
    cuda_para->w = UPTK_para->w;
    cuda_para->x = UPTK_para->x;
    cuda_para->y = UPTK_para->y;
    cuda_para->z = UPTK_para->z;
}

void cudaChannelFormatDescToUPTKChannelFormatDesc(const cudaChannelFormatDesc * cuda_para, struct UPTKChannelFormatDesc * UPTK_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    UPTK_para->f = cudaChannelFormatKindToUPTKChannelFormatKind(cuda_para->f);
    UPTK_para->w = cuda_para->w;
    UPTK_para->x = cuda_para->x;
    UPTK_para->y = cuda_para->y;
    UPTK_para->z = cuda_para->z;
}
// TODO: UPTKMemAccessDesc generally appears in the form of an array, so special conversion is performed here.
cudaMemAccessDesc UPTKMemAccessDescTocudaMemAccessDesc(struct UPTKMemAccessDesc UPTK_para)
{
    // if (nullptr == UPTK_para) {
    //     fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
    //     abort();
    // }
    cudaMemAccessDesc cuda_para;
    cuda_para.flags = UPTKMemAccessFlagsTocudaMemAccessFlags(UPTK_para.flags);
    UPTKMemLocationTocudaMemLocation(&(UPTK_para.location), &(cuda_para.location));
    return cuda_para;
}

UPTKMemAccessDesc cudaMemAccessDescTUPTKMemAccessDesc(struct cudaMemAccessDesc cuda_para)
{
    // if (nullptr == UPTK_para) {
    //     fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
    //     abort();
    // }
    UPTKMemAccessDesc UPTK_para;
    UPTK_para.flags = cudaMemAccessFlagsToUPTKMemAccessFlags(cuda_para.flags);
    cudaMemLocationToUPTKMemLocation(&(cuda_para.location), &(UPTK_para.location));
    return UPTK_para;
}

void UPTKExternalMemoryBufferDescTocudaExternalMemoryBufferDesc(const struct UPTKExternalMemoryBufferDesc * UPTK_para, cudaExternalMemoryBufferDesc * cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->flags = UPTK_para->flags;
    cuda_para->offset = UPTK_para->offset;
    cuda_para->size = UPTK_para->size;
}

void UPTKExternalMemoryHandleDescTocudaExternalMemoryHandleDesc(const struct UPTKExternalMemoryHandleDesc * UPTK_para, cudaExternalMemoryHandleDesc * cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->flags = UPTK_para->flags;
    memcpy(&(cuda_para->handle), &(UPTK_para->handle), sizeof(cuda_para->handle));
    cuda_para->size = UPTK_para->size;
    cuda_para->type = UPTKExternalMemoryHandleTypeTocudaExternalMemoryHandleType(UPTK_para->type);
}

void UPTKExternalSemaphoreHandleDescTocudaExternalSemaphoreHandleDesc(const struct UPTKExternalSemaphoreHandleDesc * UPTK_para, cudaExternalSemaphoreHandleDesc * cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->flags = UPTK_para->flags;
    memcpy(&(cuda_para->handle), &(UPTK_para->handle), sizeof(cuda_para->handle));
    cuda_para->type = UPTKExternalSemaphoreHandleTypeTocudaExternalSemaphoreHandleType(UPTK_para->type);
}

// TODO: UPTKExternalSemaphoreSignalParams generally appears in the form of an array, so special conversion is performed here.
cudaExternalSemaphoreSignalParams UPTKExternalSemaphoreSignalParamsTocudaExternalSemaphoreSignalParams(struct UPTKExternalSemaphoreSignalParams UPTK_para)
{
    // if (nullptr == UPTK_para || nullptr == cuda_para) {
    //     fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
    //     abort();
    // }
    cudaExternalSemaphoreSignalParams cuda_para;
    cuda_para.flags = UPTK_para.flags;
    memcpy(cuda_para.reserved, UPTK_para.reserved, sizeof(cuda_para.reserved));
    //Copy the members of the params structure
    cuda_para.params.fence.value = UPTK_para.params.fence.value;
    cuda_para.params.keyedMutex.key = UPTK_para.params.keyedMutex.key;
    memcpy(cuda_para.params.reserved, UPTK_para.params.reserved, sizeof(cuda_para.params.reserved));
    memcpy(&cuda_para.params.nvSciSync, &UPTK_para.params.nvSciSync, sizeof(cuda_para.params.nvSciSync));
    return cuda_para;
}

// TODO: UPTKExternalSemaphoreSignalParams generally appears in the form of an array, so special conversion is performed here.
UPTKExternalSemaphoreSignalParams cudaExternalSemaphoreSignalParamsToUPTKExternalSemaphoreSignalParams(cudaExternalSemaphoreSignalParams cuda_para)
{
    // if (nullptr == UPTK_para || nullptr == cuda_para) {
    //     fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
    //     abort();
    // }
    UPTKExternalSemaphoreSignalParams UPTK_para;
    UPTK_para.flags = cuda_para.flags;
    memcpy(UPTK_para.reserved, cuda_para.reserved, sizeof(UPTK_para.reserved));
    //Copy the members of the params structure
    UPTK_para.params.fence.value = cuda_para.params.fence.value;
    UPTK_para.params.keyedMutex.key = cuda_para.params.keyedMutex.key;
    memcpy(UPTK_para.params.reserved, cuda_para.params.reserved, sizeof(UPTK_para.params.reserved));
    memcpy(&UPTK_para.params.nvSciSync, &cuda_para.params.nvSciSync, sizeof(UPTK_para.params.nvSciSync));
    return UPTK_para;
}

//TODO: UPTKExternalSemaphoreWaitParams generally appears in the form of an array, so special conversion is performed here.
cudaExternalSemaphoreWaitParams UPTKExternalSemaphoreWaitParamsTocudaExternalSemaphoreWaitParams(struct UPTKExternalSemaphoreWaitParams UPTK_para)
{
    // if (nullptr == UPTK_para || nullptr == cuda_para) {
    //     fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
    //     abort();
    // }
    cudaExternalSemaphoreWaitParams cuda_para;
    cuda_para.flags = UPTK_para.flags;
    memcpy(cuda_para.reserved, UPTK_para.reserved, sizeof(cuda_para.reserved));
    //Copy the members of the params structure
    cuda_para.params.fence.value = UPTK_para.params.fence.value;
    cuda_para.params.keyedMutex.key = UPTK_para.params.keyedMutex.key;
    cuda_para.params.keyedMutex.timeoutMs = UPTK_para.params.keyedMutex.timeoutMs;
    memcpy(cuda_para.params.reserved, UPTK_para.params.reserved, sizeof(cuda_para.params.reserved));
    memcpy(&cuda_para.params.nvSciSync, &UPTK_para.params.nvSciSync, sizeof(cuda_para.params.nvSciSync));
    return cuda_para;
}

//TODO: UPTKExternalSemaphoreWaitParams generally appears in the form of an array, so special conversion is performed here.
UPTKExternalSemaphoreWaitParams cudaExternalSemaphoreWaitParamsToUPTKExternalSemaphoreWaitParams(cudaExternalSemaphoreWaitParams cuda_para)
{
    // if (nullptr == UPTK_para || nullptr == cuda_para) {
    //     fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
    //     abort();
    // }
    UPTKExternalSemaphoreWaitParams UPTK_para;
    UPTK_para.flags = cuda_para.flags;
    memcpy(UPTK_para.reserved, cuda_para.reserved, sizeof(UPTK_para.reserved));
    //Copy the members of the params structure
    UPTK_para.params.fence.value = cuda_para.params.fence.value;
    UPTK_para.params.keyedMutex.key = cuda_para.params.keyedMutex.key;
    UPTK_para.params.keyedMutex.timeoutMs = cuda_para.params.keyedMutex.timeoutMs;
    memcpy(UPTK_para.params.reserved, cuda_para.params.reserved, sizeof(UPTK_para.params.reserved));
    memcpy(&UPTK_para.params.nvSciSync, &cuda_para.params.nvSciSync, sizeof(UPTK_para.params.nvSciSync));
    return UPTK_para;
}

void UPTKHostNodeParamsTocudaHostNodeParams(const struct UPTKHostNodeParams * UPTK_para, cudaHostNodeParams * cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->fn = (cudaHostFn_t)UPTK_para->fn;
    cuda_para->userData = UPTK_para->userData;
}

void cudaHostNodeParamsToUPTKHostNodeParams(const cudaHostNodeParams * cuda_para, struct UPTKHostNodeParams * UPTK_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    UPTK_para->fn = (UPTKHostFn_t)cuda_para->fn;
    UPTK_para->userData = cuda_para->userData;
}

void UPTKMemLocationTocudaMemLocation(const struct UPTKMemLocation * UPTK_para, cudaMemLocation * cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->id = UPTK_para->id;
    cuda_para->type = UPTKMemLocationTypeTocudaMemLocationType(UPTK_para->type);
}

void cudaMemLocationToUPTKMemLocation(const struct cudaMemLocation * cuda_para, UPTKMemLocation * UPTK_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    UPTK_para->id = cuda_para->id;
    UPTK_para->type = cudaMemLocationTypeToUPTKMemLocationType(cuda_para->type);
}

void UPTKPitchedPtrTocudaPitchedPtr(const struct UPTKPitchedPtr * UPTK_para, cudaPitchedPtr * cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->pitch = UPTK_para->pitch;
    cuda_para->ptr = UPTK_para->ptr;
    cuda_para->xsize = UPTK_para->xsize;
    cuda_para->ysize = UPTK_para->ysize;
}

void cudaPitchedPtrToUPTKPitchedPtr(const cudaPitchedPtr * cuda_para, struct UPTKPitchedPtr * UPTK_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    UPTK_para->pitch = cuda_para->pitch;
    UPTK_para->ptr = cuda_para->ptr;
    UPTK_para->xsize = cuda_para->xsize;
    UPTK_para->ysize = cuda_para->ysize;
}

void UPTKResourceDescTocudaResourceDesc(const struct UPTKResourceDesc * UPTK_para, cudaResourceDesc * cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    memcpy(&(cuda_para->res), &(UPTK_para->res), sizeof(UPTK_para->res));
    cuda_para->resType = UPTKResourceTypeTocudaResourceType(UPTK_para->resType);
}

void cudaResourceDescToUPTKResourceDesc(const cudaResourceDesc * cuda_para, struct UPTKResourceDesc * UPTK_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    memcpy(&(UPTK_para->res), &(cuda_para->res), sizeof(cuda_para->res));
    UPTK_para->resType = cudaResourceTypeToUPTKResourceType(cuda_para->resType);
}

void UPTKIpcEventHandleTocudaIpcEventHandle(const UPTKIpcEventHandle_t * UPTK_para, cudaIpcEventHandle_t * cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    int len = min(UPTK_IPC_HANDLE_SIZE, CUDA_IPC_HANDLE_SIZE);
    for (int i = 0; i < len; ++i) {
        cuda_para->reserved[i] = UPTK_para->reserved[i];
    }
}

void cudaIpcEventHandleToUPTKIpcEventHandle(const cudaIpcEventHandle_t * cuda_para, UPTKIpcEventHandle_t * UPTK_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    int len = min(UPTK_IPC_HANDLE_SIZE, CUDA_IPC_HANDLE_SIZE);
    for (int i = 0; i < len; ++i) {
        UPTK_para->reserved[i] = cuda_para->reserved[i];
    }
}

void UPTKIpcMemHandleTocudaIpcMemHandle(const UPTKIpcMemHandle_t * UPTK_para, cudaIpcMemHandle_t * cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    int len = min(UPTK_IPC_HANDLE_SIZE, CUDA_IPC_HANDLE_SIZE);
    for (int i = 0; i < len; ++i) {
        cuda_para->reserved[i] = UPTK_para->reserved[i];
    }
}

void cudaIpcMemHandleToUPTKIpcMemHandle(const cudaIpcMemHandle_t * cuda_para, UPTKIpcMemHandle_t * UPTK_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    int len = min(UPTK_IPC_HANDLE_SIZE, CUDA_IPC_HANDLE_SIZE);
    for (int i = 0; i < len; ++i) {
        UPTK_para->reserved[i] = cuda_para->reserved[i];
    }
}

void UPTKPosTocudaPos(const struct UPTKPos * UPTK_para, cudaPos * cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->x = UPTK_para->x;
    cuda_para->y = UPTK_para->y;
    cuda_para->z = UPTK_para->z;
}

void cudaPosToUPTKPos(const cudaPos * cuda_para, struct UPTKPos * UPTK_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    UPTK_para->x = cuda_para->x;
    UPTK_para->y = cuda_para->y;
    UPTK_para->z = cuda_para->z;
}

void UPTKKernelNodeAttrValueTocudaKernelNodeAttrValue(const UPTKKernelNodeAttrValue *UPTK_para, cudaKernelNodeAttrValue *cuda_para, UPTKKernelNodeAttrID attr)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    //The union structure needs to determine the specific union type selection according to the value of other variables.
    //There, the CUkernelNodeAttrID variable determines the type of the CUkernelNodeAttrValue union, and is compatible with the cuda structure. There are only two optional types.
    if (attr == UPTKKernelNodeAttributeCooperative) {
        cuda_para->cooperative =UPTK_para->cooperative;
    }
    else if (attr == UPTKKernelNodeAttributeAccessPolicyWindow) {
        UPTKAccessPolicyWindowTocudaAccessPolicyWindow(&(UPTK_para->accessPolicyWindow), &(cuda_para->accessPolicyWindow));
    }
}

void cudaKernelNodeAttrValueToUPTKKernelNodeAttrValue(const cudaKernelNodeAttrValue *cuda_para, UPTKKernelNodeAttrValue *UPTK_para, UPTKKernelNodeAttrID attr)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    //The union structure needs to determine the specific union type selection according to the value of other variables.
    //There, the CUkernelNodeAttrID variable determines the type of the CUkernelNodeAttrValue union, and is compatible with the cuda structure. There are only two optional types.
    if (attr == UPTKKernelNodeAttributeCooperative) {
        UPTK_para->cooperative = cuda_para->cooperative;
    }
    else if (attr == UPTKKernelNodeAttributeAccessPolicyWindow) {
        cudaAccessPolicyWindowToUPTKAccessPolicyWindow(&(cuda_para->accessPolicyWindow), &(UPTK_para->accessPolicyWindow));
    }
}

void UPTKMemPoolPropsTocudaMemPoolProps(const UPTKMemPoolProps * UPTK_para, cudaMemPoolProps * cuda_para) {
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    memcpy(cuda_para->reserved, UPTK_para->reserved, sizeof(cuda_para->reserved));
    cuda_para->win32SecurityAttributes = UPTK_para->win32SecurityAttributes;
    cuda_para->allocType = UPTKMemAllocationTypeTocudaMemAllocationType(UPTK_para->allocType);
    cuda_para->handleTypes = UPTKMemAllocationHandleTypeTocudaMemAllocationHandleType(UPTK_para->handleTypes);
    UPTKMemLocationTocudaMemLocation(&(UPTK_para->location), &(cuda_para->location));
}

void cudaMemPoolPropsToUPTKMemPoolProps(const cudaMemPoolProps * cuda_para, UPTKMemPoolProps * UPTK_para) {
    if (nullptr == UPTK_para || nullptr == cuda_para)
    {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    memcpy(UPTK_para->reserved, cuda_para->reserved, sizeof(UPTK_para->reserved));
    UPTK_para->win32SecurityAttributes = cuda_para->win32SecurityAttributes;
    UPTK_para->allocType = cudaMemAllocationTypeToUPTKMemAllocationType(cuda_para->allocType);
    UPTK_para->handleTypes = cudaMemAllocationHandleTypeToUPTKMemAllocationHandleType(cuda_para->handleTypes);
    cudaMemLocationToUPTKMemLocation(&(cuda_para->location), &(UPTK_para->location));
}

cudaGraphMemAttributeType UPTKGraphMemAttributeTypeTocudaGraphMemAttributeType(enum UPTKGraphMemAttributeType para) {
        switch (para) {
        case UPTKGraphMemAttrUsedMemCurrent:
            return cudaGraphMemAttrUsedMemCurrent;
        case UPTKGraphMemAttrUsedMemHigh:
            return cudaGraphMemAttrUsedMemHigh;
        case UPTKGraphMemAttrReservedMemCurrent:
            return cudaGraphMemAttrReservedMemCurrent;
        case UPTKGraphMemAttrReservedMemHigh:
            return cudaGraphMemAttrReservedMemHigh;
        default:
            ERROR_INVALID_ENUM(); 
        }
}

void UPTKMemAllocNodeParamsTocudaMemAllocNodeParams(const struct UPTKMemAllocNodeParams *UPTK_para, cudaMemAllocNodeParams *cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    UPTKMemPoolPropsTocudaMemPoolProps(&(UPTK_para->poolProps),&(cuda_para->poolProps));

    cuda_para->accessDescCount = UPTK_para->accessDescCount;
    cudaMemAccessDesc* cuda_daccessDescs = (cudaMemAccessDesc*)malloc(sizeof(cudaMemAccessDesc) * cuda_para->accessDescCount);
    for (int i = 0; i < cuda_para->accessDescCount; i++)
    {
        cuda_daccessDescs[i] = UPTKMemAccessDescTocudaMemAccessDesc(UPTK_para->accessDescs[i]);
    }
    cuda_para->bytesize = UPTK_para->bytesize;
    cuda_para->dptr = UPTK_para->dptr;
}

void cudaMemAllocNodeParamsToUPTKMemAllocNodeParams(const struct cudaMemAllocNodeParams *cuda_para, UPTKMemAllocNodeParams *UPTK_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cudaMemPoolPropsToUPTKMemPoolProps(&(cuda_para->poolProps), &(UPTK_para->poolProps));

    UPTK_para->accessDescCount = cuda_para->accessDescCount;
    UPTKMemAccessDesc* UPTK_daccessDescs = (UPTKMemAccessDesc*)malloc(sizeof(UPTKMemAccessDesc) * UPTK_para->accessDescCount);
    for (int i = 0; i < UPTK_para->accessDescCount; i++)
    {
        UPTK_daccessDescs[i] = cudaMemAccessDescTUPTKMemAccessDesc(cuda_para->accessDescs[i]);
    }
    UPTK_para->bytesize = cuda_para->bytesize;
    UPTK_para->dptr = cuda_para->dptr;
}

cudaGLDeviceList UPTKGLDeviceListTocudaGLDeviceList(UPTKGLDeviceList para) {
    switch (para) {
        case UPTKGLDeviceListAll:
            return cudaGLDeviceListAll;
        case UPTKGLDeviceListCurrentFrame:
            return cudaGLDeviceListCurrentFrame;
        case UPTKGLDeviceListNextFrame:
            return cudaGLDeviceListNextFrame;
        default:
            ERROR_INVALID_ENUM();
    }
}

void UPTKExternalMemoryMipmappedArrayDescTocudaExternalMemoryMipmappedArrayDesc(const UPTKExternalMemoryMipmappedArrayDesc * UPTK_para, cudaExternalMemoryMipmappedArrayDesc * cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->offset = UPTK_para->offset;
    UPTKChannelFormatDescTocudaChannelFormatDesc(&(UPTK_para->formatDesc), &(cuda_para->formatDesc));
    UPTKExtentTocudaExtent(&(UPTK_para->extent), &(cuda_para->extent));
    cuda_para->flags = UPTK_para->flags;
    cuda_para->numLevels = UPTK_para->numLevels;
}

cudaExternalSemaphoreSignalNodeParams UPTKExternalSemaphoreSignalNodeParamsTocudaExternalSemaphoreSignalNodeParams(UPTKExternalSemaphoreSignalNodeParams UPTK_para)
{
    // if (nullptr == UPTK_para || nullptr == cuda_para) {
    //     fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
    //     abort();
    // }
    cudaExternalSemaphoreSignalNodeParams cuda_para;
    cuda_para.extSemArray = (cudaExternalSemaphore_t *)UPTK_para.extSemArray;
    cuda_para.numExtSems = UPTK_para.numExtSems;
    cudaExternalSemaphoreSignalParams *cuda_paramsArray = (cudaExternalSemaphoreSignalParams *)malloc(sizeof(cudaExternalSemaphoreSignalParams) * cuda_para.numExtSems);
    for (int i = 0; i < cuda_para.numExtSems; i++){
        cuda_paramsArray[i] = UPTKExternalSemaphoreSignalParamsTocudaExternalSemaphoreSignalParams(UPTK_para.paramsArray[i]);
    }
    cuda_para.paramsArray = cuda_paramsArray;
    return cuda_para;
}

UPTKExternalSemaphoreSignalNodeParams cudaExternalSemaphoreSignalNodeParamsToUPTKExternalSemaphoreSignalNodeParams(cudaExternalSemaphoreSignalNodeParams cuda_para)
{
    // if (nullptr == UPTK_para || nullptr == cuda_para) {
    //     fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
    //     abort();
    // }
    UPTKExternalSemaphoreSignalNodeParams UPTK_para;
    UPTK_para.extSemArray = (UPTKExternalSemaphore_t *)cuda_para.extSemArray;
    UPTK_para.numExtSems = cuda_para.numExtSems;
    UPTKExternalSemaphoreSignalParams *UPTK_paramsArray = (UPTKExternalSemaphoreSignalParams *)malloc(sizeof(UPTKExternalSemaphoreSignalParams) * UPTK_para.numExtSems);
    for (int i = 0; i < UPTK_para.numExtSems; i++){
        UPTK_paramsArray[i] = cudaExternalSemaphoreSignalParamsToUPTKExternalSemaphoreSignalParams(cuda_para.paramsArray[i]);
    }
    UPTK_para.paramsArray = UPTK_paramsArray;
    return UPTK_para;
}

cudaExternalSemaphoreWaitNodeParams UPTKExternalSemaphoreWaitNodeParamsTocudaExternalSemaphoreWaitNodeParams(UPTKExternalSemaphoreWaitNodeParams UPTK_para)
{
    // if (nullptr == UPTK_para || nullptr == cuda_para) {
    //     fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
    //     abort();
    // }
    cudaExternalSemaphoreWaitNodeParams cuda_para;
    cuda_para.extSemArray = (cudaExternalSemaphore_t *)UPTK_para.extSemArray;
    cuda_para.numExtSems = UPTK_para.numExtSems;
    cudaExternalSemaphoreWaitParams * cuda_paramsArray = (cudaExternalSemaphoreWaitParams *)malloc(sizeof(cudaExternalSemaphoreWaitParams) * cuda_para.numExtSems);
    for (int i = 0; i < cuda_para.numExtSems; i++){
        cuda_paramsArray[i] = UPTKExternalSemaphoreWaitParamsTocudaExternalSemaphoreWaitParams(UPTK_para.paramsArray[i]);
    }
    cuda_para.paramsArray = cuda_paramsArray;
    return cuda_para;
}

UPTKExternalSemaphoreWaitNodeParams cudaExternalSemaphoreWaitNodeParamsToUPTKExternalSemaphoreWaitNodeParams(cudaExternalSemaphoreWaitNodeParams cuda_para)
{
    // if (nullptr == UPTK_para || nullptr == cuda_para) {
    //     fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
    //     abort();
    // }
    UPTKExternalSemaphoreWaitNodeParams UPTK_para;
    UPTK_para.extSemArray = (UPTKExternalSemaphore_t *)cuda_para.extSemArray;
    UPTK_para.numExtSems = cuda_para.numExtSems;
    UPTKExternalSemaphoreWaitParams * UPTK_paramsArray = (UPTKExternalSemaphoreWaitParams *)malloc(sizeof(UPTKExternalSemaphoreWaitParams) * UPTK_para.numExtSems);
    for (int i = 0; i < UPTK_para.numExtSems; i++){
        UPTK_paramsArray[i] = cudaExternalSemaphoreWaitParamsToUPTKExternalSemaphoreWaitParams(cuda_para.paramsArray[i]);
    }
    UPTK_para.paramsArray = UPTK_paramsArray;
    return UPTK_para;
}


void UPTKMemcpy3DPeerParmsTocudaMemcpy3DPeerParms(const struct UPTKMemcpy3DPeerParms * UPTK_para, cudaMemcpy3DPeerParms * cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->srcArray = (cudaArray_t)UPTK_para->srcArray;
    UPTKPosTocudaPos(&(UPTK_para->srcPos), &(cuda_para->srcPos));
    UPTKPitchedPtrTocudaPitchedPtr(&(UPTK_para->srcPtr), &(cuda_para->srcPtr));
    cuda_para->srcDevice = UPTK_para->srcDevice;
    cuda_para->dstArray = (cudaArray_t)UPTK_para->dstArray;
    UPTKPosTocudaPos(&(UPTK_para->dstPos), &(cuda_para->dstPos));
    UPTKPitchedPtrTocudaPitchedPtr(&(UPTK_para->dstPtr), &(cuda_para->dstPtr));
    cuda_para->dstDevice = UPTK_para->dstDevice;
    UPTKExtentTocudaExtent(&(UPTK_para->extent), &(cuda_para->extent));
}

void UPTKArrayMemoryRequirementsTocudaArrayMemoryRequirements(const struct UPTKArrayMemoryRequirements * UPTK_para, cudaArrayMemoryRequirements * cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->alignment = UPTK_para->alignment;
    cuda_para->size = UPTK_para->size;
    memcpy(cuda_para->reserved, UPTK_para->reserved, sizeof(cuda_para->reserved));
}

void cudaArrayMemoryRequirementsToUPTKArrayMemoryRequirements(const struct cudaArrayMemoryRequirements * cuda_para, UPTKArrayMemoryRequirements * UPTK_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    UPTK_para->alignment = cuda_para->alignment;
    UPTK_para->size = cuda_para->size;
    memcpy(UPTK_para->reserved, cuda_para->reserved, sizeof(UPTK_para->reserved));
}

void UPTKLaunchConfig_tTocudaLaunchConfig_t(const UPTKLaunchConfig_t * UPTK_para, cudaLaunchConfig_t * cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->gridDim = UPTK_para->gridDim;
    cuda_para->blockDim = UPTK_para->blockDim;
    cuda_para->dynamicSmemBytes = UPTK_para->dynamicSmemBytes;
    cuda_para->stream = (cudaStream_t)UPTK_para->stream;

    cuda_para->numAttrs = UPTK_para->numAttrs;
    cudaLaunchAttribute* cuda_attrs = (cudaLaunchAttribute*)malloc(sizeof(cudaLaunchAttribute) * cuda_para->numAttrs);
    // attrs is a nullable pointer. if numAttrs == 0, attrs can be nullptr.
    if (UPTK_para->attrs != nullptr)
    {
        for (int i = 0; i < cuda_para->numAttrs; i++)
        {
            cuda_attrs[i] = UPTKLaunchAttributeTocudaLaunchAttribute(UPTK_para->attrs[i]);
        }
    }
    cuda_para->attrs = cuda_attrs;
}
// TODO: UPTKLaunchAttribute generally appears in the form of an array, so special conversion is performed here.
cudaLaunchAttribute UPTKLaunchAttributeTocudaLaunchAttribute(UPTKLaunchAttribute UPTK_para)
{
    // if (nullptr == UPTK_para || nullptr == cuda_para) {
    //     fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
    //     abort();
    // }
    cudaLaunchAttribute cuda_para;
    cuda_para.id = UPTKLaunchAttributeIDTocudaLaunchAttributeID(UPTK_para.id);
    UPTKLaunchAttributeValueTocudaLaunchAttributeValue(&(UPTK_para.val), &(cuda_para.val), UPTK_para.id);
    // Note: Padding in UPTKLaunchAttribute is ignored during conversion
    return cuda_para;
}

cudaLaunchAttributeID UPTKLaunchAttributeIDTocudaLaunchAttributeID(enum UPTKLaunchAttributeID para)
{
    switch(para) {
        case UPTKLaunchAttributeCooperative:
            return cudaLaunchAttributeCooperative;
        case UPTKLaunchAttributeSynchronizationPolicy:
            return cudaLaunchAttributeSynchronizationPolicy;
        case UPTKLaunchAttributeIgnore:
            return cudaLaunchAttributeIgnore;
        case UPTKLaunchAttributeAccessPolicyWindow:
            return cudaLaunchAttributeAccessPolicyWindow;
        case UPTKLaunchAttributeClusterDimension:
            return cudaLaunchAttributeClusterDimension;
        case UPTKLaunchAttributeClusterSchedulingPolicyPreference:
            return cudaLaunchAttributeClusterSchedulingPolicyPreference;
        case UPTKLaunchAttributeProgrammaticStreamSerialization:
            return cudaLaunchAttributeProgrammaticStreamSerialization;
        case UPTKLaunchAttributeProgrammaticEvent:
            return cudaLaunchAttributeProgrammaticEvent;
        case UPTKLaunchAttributePriority:
            return cudaLaunchAttributePriority;
        default:
            ERROR_INVALID_ENUM();                 
    }
}

void UPTKLaunchAttributeValueTocudaLaunchAttributeValue(const UPTKLaunchAttributeValue *UPTK_para, cudaLaunchAttributeValue *cuda_para, UPTKLaunchAttributeID para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para)
    {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    switch (para)
    {
    case UPTKLaunchAttributeCooperative: // cuda underlying hardware support
        cuda_para->cooperative = UPTK_para->cooperative;
        break;
    case UPTKLaunchAttributeAccessPolicyWindow: // cuda underlying hardware support
        UPTKAccessPolicyWindowTocudaAccessPolicyWindow(&(UPTK_para->accessPolicyWindow), &(cuda_para->accessPolicyWindow));
        break;
    case UPTKLaunchAttributeIgnore:
        memcpy(cuda_para->pad, UPTK_para->pad, sizeof(cuda_para->pad));
        break;
    case UPTKLaunchAttributeSynchronizationPolicy:
        cuda_para->syncPolicy = UPTKSynchronizationPolicyTocudaSynchronizationPolicy(UPTK_para->syncPolicy);
        break;
    case UPTKLaunchAttributeClusterDimension:
        memcpy(&(cuda_para->clusterDim), &(UPTK_para->clusterDim), sizeof(cuda_para->clusterDim));
        break;
    case UPTKLaunchAttributeClusterSchedulingPolicyPreference:
        cuda_para->clusterSchedulingPolicyPreference = UPTKClusterSchedulingPolicyTocudaClusterSchedulingPolicy(UPTK_para->clusterSchedulingPolicyPreference);
        break;
    case UPTKLaunchAttributeProgrammaticStreamSerialization:
        cuda_para->programmaticStreamSerializationAllowed = UPTK_para->programmaticStreamSerializationAllowed;
        break;
    case UPTKLaunchAttributeProgrammaticEvent:
        cuda_para->programmaticEvent.event = (cudaEvent_t)UPTK_para->programmaticEvent.event;
        cuda_para->programmaticEvent.flags = UPTK_para->programmaticEvent.flags;
        cuda_para->programmaticEvent.triggerAtBlockStart = UPTK_para->programmaticEvent.triggerAtBlockStart;
        break;
    case UPTKLaunchAttributePriority:
        cuda_para->priority = UPTK_para->priority;
        break;
    default:
        ERROR_INVALID_ENUM();
        break;
    }
}

void cudaLaunchAttributeValueToUPTKLaunchAttributeValue(const cudaLaunchAttributeValue * cuda_para, UPTKLaunchAttributeValue * UPTK_para, cudaLaunchAttributeID para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    switch (para)
    {
    case cudaLaunchAttributeCooperative: // cuda underlying hardware support
        UPTK_para->cooperative = cuda_para->cooperative;
        break;
    case cudaLaunchAttributeAccessPolicyWindow: // cuda underlying hardware support
        cudaAccessPolicyWindowToUPTKAccessPolicyWindow(&(cuda_para->accessPolicyWindow), &(UPTK_para->accessPolicyWindow));
        break;
    case cudaLaunchAttributeIgnore:
        memcpy(UPTK_para->pad, cuda_para->pad, sizeof(UPTK_para->pad));
        break;
    case cudaLaunchAttributeSynchronizationPolicy:      // cuda underlying hardware support
        UPTK_para->syncPolicy = cudaSynchronizationPolicyToUPTKSynchronizationPolicy(cuda_para->syncPolicy);
        break;
    case cudaLaunchAttributeClusterDimension:
        memcpy(&(UPTK_para->clusterDim), &(cuda_para->clusterDim), sizeof(UPTK_para->clusterDim));
        break;
    case cudaLaunchAttributeClusterSchedulingPolicyPreference:
        UPTK_para->clusterSchedulingPolicyPreference = cudaClusterSchedulingPolicyToUPTKClusterSchedulingPolicy(cuda_para->clusterSchedulingPolicyPreference);
        break;
    case cudaLaunchAttributeProgrammaticStreamSerialization:
        UPTK_para->programmaticStreamSerializationAllowed = cuda_para->programmaticStreamSerializationAllowed;
        break;
    case cudaLaunchAttributeProgrammaticEvent:
        UPTK_para->programmaticEvent.event = (UPTKEvent_t)cuda_para->programmaticEvent.event;
        UPTK_para->programmaticEvent.flags = cuda_para->programmaticEvent.flags;
        UPTK_para->programmaticEvent.triggerAtBlockStart = cuda_para->programmaticEvent.triggerAtBlockStart;
        break;
    case cudaLaunchAttributePriority:
        UPTK_para->priority = cuda_para->priority;
        break;
    default:
        ERROR_INVALID_ENUM();
        break;
    }
}

cudaClusterSchedulingPolicy UPTKClusterSchedulingPolicyTocudaClusterSchedulingPolicy(UPTKClusterSchedulingPolicy para)
{
    switch(para){
        case UPTKClusterSchedulingPolicyDefault:
            return cudaClusterSchedulingPolicyDefault;
        case UPTKClusterSchedulingPolicySpread:
            return cudaClusterSchedulingPolicySpread;
        case UPTKClusterSchedulingPolicyLoadBalancing:
            return cudaClusterSchedulingPolicyLoadBalancing;
        default:
            ERROR_INVALID_ENUM();
    }
}

UPTKClusterSchedulingPolicy cudaClusterSchedulingPolicyToUPTKClusterSchedulingPolicy(cudaClusterSchedulingPolicy para) {
    switch(para){
        case cudaClusterSchedulingPolicyDefault:
            return UPTKClusterSchedulingPolicyDefault;
        case cudaClusterSchedulingPolicySpread:
            return UPTKClusterSchedulingPolicySpread;
        case cudaClusterSchedulingPolicyLoadBalancing:
            return UPTKClusterSchedulingPolicyLoadBalancing;
        default:
            ERROR_INVALID_ENUM();
    }
}

enum UPTKSynchronizationPolicy cudaSynchronizationPolicyToUPTKSynchronizationPolicy(enum cudaSynchronizationPolicy para) {
    switch (para) {
        case cudaSyncPolicyAuto:
            return UPTKSyncPolicyAuto;
        case cudaSyncPolicyBlockingSync:
            return UPTKSyncPolicyBlockingSync;
        case cudaSyncPolicySpin:
            return UPTKSyncPolicySpin;
        case cudaSyncPolicyYield:
            return UPTKSyncPolicyYield;
        default:
            ERROR_INVALID_ENUM();
    }
}

enum UPTKTextureAddressMode cudaTextureAddressModeToUPTKTextureAddressMode(enum cudaTextureAddressMode para) {
    switch (para) {
        case cudaAddressModeBorder:
            return UPTKAddressModeBorder;
        case cudaAddressModeClamp:
            return UPTKAddressModeClamp;
        case cudaAddressModeMirror:
            return UPTKAddressModeMirror;
        case cudaAddressModeWrap:
            return UPTKAddressModeWrap;
        default:
            ERROR_INVALID_ENUM();
    }
}

enum UPTKTextureFilterMode cudaTextureFilterModeToUPTKTextureFilterMode(enum cudaTextureFilterMode para) {
    switch (para) {
        case cudaFilterModeLinear:
            return UPTKFilterModeLinear;
        case cudaFilterModePoint:
            return UPTKFilterModePoint;
        default:
            ERROR_INVALID_ENUM();
    }
}

enum UPTKTextureReadMode cudaTextureReadModeToUPTKTextureReadMode(enum cudaTextureReadMode para) {
    switch (para) {
        case cudaReadModeElementType:
            return UPTKReadModeElementType;
        case cudaReadModeNormalizedFloat:
            return UPTKReadModeNormalizedFloat;
        default:
            ERROR_INVALID_ENUM();
   }
}

/*void UPTKGraphNodeParamsTocudaGraphNodeParams(const UPTKGraphNodeParams * UPTK_para, cudaGraphNodeParams * cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para)
    {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->type = UPTKGraphNodeTypeTocudaGraphNodeType(UPTK_para->type);
    memcpy(cuda_para->reserved0, UPTK_para->reserved0, sizeof(int[3]));
    cuda_para->reserved2 = UPTK_para->reserved2;
    memset(&(cuda_para->reserved1), 0, sizeof(long long[29]));
    memcpy(&(cuda_para->reserved1), &(UPTK_para->reserved1), sizeof(long long[29]));
    
    switch (cuda_para->type)
    {
    case cudaGraphNodeTypeKernel:
        UPTKKernelNodeParamsV2TocudaKernelNodeParams(&(UPTK_para->kernel), &(cuda_para->kernel));
        break;
    case cudaGraphNodeTypeMemcpy:
        UPTKMemcpyNodeParamsTocudaMemcpyNodeParams(&(UPTK_para->memcpy), &(cuda_para->memcpy));
        break;
    case cudaGraphNodeTypeMemset:
        UPTKMemsetParamsV2TocudaMemsetParams(&(UPTK_para->memset), &(cuda_para->memset));
        break;
    case cudaGraphNodeTypeHost:
        UPTKHostNodeParamsV2TocudaHostNodeParams(&(UPTK_para->host), &(cuda_para->host));
        break;
    case cudaGraphNodeTypeGraph:
        UPTKChildGraphNodeParamsTocudaChildGraphNodeParams(&(UPTK_para->graph), &(cuda_para->graph));
        break;
    case cudaGraphNodeTypeWaitEvent:
        UPTKEventWaitNodeParamsTocudaEventWaitNodeParams(&(UPTK_para->eventWait), &(cuda_para->eventWait));
        break;
    case cudaGraphNodeTypeEventRecord:
        UPTKEventRecordNodeParamsTocudaEventRecordNodeParams(&(UPTK_para->eventRecord), &(cuda_para->eventRecord));
        break;
    }
}*/

void UPTKConditionalNodeParamsTocudaConditionalNodeParams(const UPTKConditionalNodeParams *UPTK_para, cudaConditionalNodeParams *cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para)
    {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->handle = (cudaGraphConditionalHandle)UPTK_para->handle;
    cuda_para->type = UPTKConditionalNodeTypeTocudaConditionalNodeType(UPTK_para->type);
    cuda_para->size = UPTK_para->size;
    cuda_para->phGraph_out = (cudaGraph_t *)UPTK_para->phGraph_out;

}

cudaGraphConditionalNodeType UPTKConditionalNodeTypeTocudaConditionalNodeType(enum UPTKGraphConditionalNodeType para)
{
    switch (para)
    {
    case UPTKGraphCondTypeIf:
        return cudaGraphCondTypeIf;
    case UPTKGraphCondTypeWhile:
        return cudaGraphCondTypeWhile;
    default:
        ERROR_INVALID_ENUM();
    }
}

void UPTKKernelNodeParamsV2TocudaKernelNodeParams(const UPTKKernelNodeParamsV2 *UPTK_para, cudaKernelNodeParams *cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para)
    {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->func = UPTK_para->func;
    cuda_para->sharedMemBytes = UPTK_para->sharedMemBytes;
    cuda_para->kernelParams = UPTK_para->kernelParams;
    cuda_para->extra = UPTK_para->extra;
#if !defined(__cplusplus) || __cplusplus >= 201103L
    cuda_para->gridDim = UPTK_para->gridDim;
    cuda_para->blockDim = UPTK_para->blockDim;
#else
    cuda_para->gridDim.x = UPTK_para->gridDim.x;
    cuda_para->gridDim.y = UPTK_para->gridDim.y;
    cuda_para->gridDim.z = UPTK_para->gridDim.z;
    cuda_para->blockDim.x = UPTK_para->blockDim.x;
    cuda_para->blockDim.y = UPTK_para->blockDim.y;
    cuda_para->blockDim.z = UPTK_para->blockDim.z;
#endif
}

void UPTKMemcpyNodeParamsTocudaMemcpyNodeParams(const UPTKMemcpyNodeParams *UPTK_para, cudaMemcpyNodeParams *cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para)
    {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->flags = UPTK_para->flags;
    memcpy(&(cuda_para->reserved), &(UPTK_para->reserved), sizeof(int[3]));
    UPTKMemcpy3DParmsTocudaMemcpy3DParms(&(UPTK_para->copyParams), &(cuda_para->copyParams));
}

void UPTKMemsetParamsV2TocudaMemsetParams(const UPTKMemsetParamsV2 *UPTK_para, cudaMemsetParams *cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para)
    {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->dst = UPTK_para->dst;
    cuda_para->elementSize = UPTK_para->elementSize;
    cuda_para->height = UPTK_para->height;
    cuda_para->pitch = UPTK_para->pitch;
    cuda_para->value = UPTK_para->value;
    cuda_para->width = UPTK_para->width;
}

void UPTKHostNodeParamsV2TocudaHostNodeParams(const UPTKHostNodeParamsV2 *UPTK_para, cudaHostNodeParams *cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para)
    {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->fn = (cudaHostFn_t)UPTK_para->fn;
    cuda_para->userData = UPTK_para->userData;
}

void UPTKChildGraphNodeParamsTocudaChildGraphNodeParams(const UPTKChildGraphNodeParams *UPTK_para, cudaChildGraphNodeParams *cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para)
    {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->graph = (cudaGraph_t)UPTK_para->graph;
}

void UPTKEventWaitNodeParamsTocudaEventWaitNodeParams(const UPTKEventWaitNodeParams *UPTK_para, cudaEventWaitNodeParams *cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para)
    {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->event = (cudaEvent_t)UPTK_para->event;
}

void UPTKEventRecordNodeParamsTocudaEventRecordNodeParams(const UPTKEventRecordNodeParams *UPTK_para, cudaEventRecordNodeParams *cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para)
    {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->event = (cudaEvent_t)UPTK_para->event;
}

cudaExternalSemaphoreSignalNodeParams UPTKExternalSemaphoreSignalNodeParamsV2TocudaExternalSemaphoreSignalNodeParams(UPTKExternalSemaphoreSignalNodeParamsV2 UPTK_para)
{
    // if (nullptr == UPTK_para || nullptr == cuda_para) {
    //     fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
    //     abort();
    // }
    cudaExternalSemaphoreSignalNodeParams cuda_para;
    cuda_para.extSemArray = (cudaExternalSemaphore_t *)UPTK_para.extSemArray;
    cuda_para.numExtSems = UPTK_para.numExtSems;
    cudaExternalSemaphoreSignalParams *cuda_paramsArray = (cudaExternalSemaphoreSignalParams *)malloc(sizeof(cudaExternalSemaphoreSignalParams) * cuda_para.numExtSems);
    for (int i = 0; i < cuda_para.numExtSems; i++)
    {
        cuda_paramsArray[i] = UPTKExternalSemaphoreSignalParamsTocudaExternalSemaphoreSignalParams(UPTK_para.paramsArray[i]);
    }
    cuda_para.paramsArray = cuda_paramsArray;
    return cuda_para;
}

cudaExternalSemaphoreWaitNodeParams UPTKExternalSemaphoreWaitNodeParamsV2TocudaExternalSemaphoreWaitNodeParams(UPTKExternalSemaphoreWaitNodeParamsV2 UPTK_para)
{
    // if (nullptr == UPTK_para || nullptr == cuda_para) {
    //     fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
    //     abort();
    // }
    cudaExternalSemaphoreWaitNodeParams cuda_para;
    cuda_para.extSemArray = (cudaExternalSemaphore_t *)UPTK_para.extSemArray;
    cuda_para.numExtSems = UPTK_para.numExtSems;
    cudaExternalSemaphoreWaitParams *cuda_paramsArray = (cudaExternalSemaphoreWaitParams *)malloc(sizeof(cudaExternalSemaphoreWaitParams) * cuda_para.numExtSems);
    for (int i = 0; i < cuda_para.numExtSems; i++)
    {
        cuda_paramsArray[i] = UPTKExternalSemaphoreWaitParamsTocudaExternalSemaphoreWaitParams(UPTK_para.paramsArray[i]);
    }
    cuda_para.paramsArray = cuda_paramsArray;
    return cuda_para;
}

void UPTKMemAllocNodeParamsV2TocudaMemAllocNodeParams(const struct UPTKMemAllocNodeParamsV2 *UPTK_para, cudaMemAllocNodeParams *cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para)
    {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    UPTKMemPoolPropsTocudaMemPoolProps(&(UPTK_para->poolProps), &(cuda_para->poolProps));

    cuda_para->accessDescCount = UPTK_para->accessDescCount;
    cudaMemAccessDesc *cuda_daccessDescs = (cudaMemAccessDesc *)malloc(sizeof(cudaMemAccessDesc) * cuda_para->accessDescCount);
    for (int i = 0; i < cuda_para->accessDescCount; i++)
    {
        cuda_daccessDescs[i] = UPTKMemAccessDescTocudaMemAccessDesc(UPTK_para->accessDescs[i]);
    }
    cuda_para->bytesize = UPTK_para->bytesize;
    cuda_para->dptr = UPTK_para->dptr;
}

void UPTKMemFreeNodeParamsTocudaMemFreeNodeParams(const UPTKMemFreeNodeParams *UPTK_para, cudaMemFreeNodeParams *cuda_para)
{
    if (nullptr == UPTK_para || nullptr == cuda_para)
    {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->dptr = UPTK_para->dptr;
}

void UPTKGraphEdgeDataTocudaGraphEdgeData(const UPTKGraphEdgeData* UPTK_para, cudaGraphEdgeData* cuda_para){
    if (nullptr == UPTK_para || nullptr == cuda_para)
    {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->from_port = UPTK_para->from_port;
    cuda_para->to_port = UPTK_para->to_port;
    cuda_para->type = UPTK_para->type;
    memcpy(&(cuda_para->reserved), &(UPTK_para->reserved), sizeof(unsigned char[5]));
}

UPTKDriverEntryPointQueryResult cudaDriverEntryPointQueryResultToUPTKDriverEntryPointQueryResult(cudaDriverEntryPointQueryResult para) {
    switch (para) {
        case cudaDriverEntryPointSuccess:
            return UPTKDriverEntryPointSuccess;
        case cudaDriverEntryPointSymbolNotFound:
            return UPTKDriverEntryPointSymbolNotFound;
        case cudaDriverEntryPointVersionNotSufficent:
            return UPTKDriverEntryPointVersionNotSufficent;
        default:
            ERROR_INVALID_ENUM();    
    }
}

void UPTKGraphInstantiateParamsTocudaGraphInstantiateParams(const UPTKGraphInstantiateParams * UPTK_para, cudaGraphInstantiateParams * cuda_para) {
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    cuda_para->flags = UPTK_para->flags;
    cuda_para->uploadStream = (cudaStream_t) UPTK_para->uploadStream;
    cuda_para->errNode_out = (cudaGraphNode_t) UPTK_para->errNode_out;
    cuda_para->result_out = UPTKGraphInstantiateResultTocudaGraphInstantiateResult(UPTK_para->result_out);
}

cudaGraphInstantiateResult UPTKGraphInstantiateResultTocudaGraphInstantiateResult(UPTKGraphInstantiateResult para) {
    switch(para) {
        case UPTKGraphInstantiateSuccess:
            return cudaGraphInstantiateSuccess;
        case UPTKGraphInstantiateError:
            return cudaGraphInstantiateError;
        case UPTKGraphInstantiateInvalidStructure:
            return cudaGraphInstantiateInvalidStructure;
        case UPTKGraphInstantiateNodeOperationNotSupported:
            return cudaGraphInstantiateNodeOperationNotSupported;
        case UPTKGraphInstantiateMultipleDevicesNotSupported:
            return cudaGraphInstantiateMultipleDevicesNotSupported;
        default:
            ERROR_INVALID_ENUM();    
    }
}

cudaMemLocation UPTKMemLocationTocudaMemLocation_v2(struct UPTKMemLocation UPTK_para)
{
    // if (nullptr == UPTK_para || nullptr == cuda_para) {
    //     fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
    //     abort();
    // }
    cudaMemLocation cuda_para;
    cuda_para.id = UPTK_para.id;
    cuda_para.type = UPTKMemLocationTypeTocudaMemLocationType(UPTK_para.type);
    return cuda_para;
}

UPTKAsyncNotificationType cudaAsyncNotificationTypeToUPTKAsyncNotificationType(cudaAsyncNotificationType para){
    switch(para) {
    case cudaAsyncNotificationTypeOverBudget:
        return UPTKAsyncNotificationTypeOverBudget;
    default:
        ERROR_INVALID_ENUM();    
    }
}

/*void cudaGraphExecUpdateResultInfoToUPTKGraphExecUpdateResultInfo(const cudaGraphExecUpdateResultInfo * cuda_para, UPTKGraphExecUpdateResultInfo * UPTK_para){
    if (nullptr == UPTK_para || nullptr == cuda_para) {
        fprintf(stderr, "%s para is nullptr\n", __FUNCTION__);
        abort();
    }
    UPTK_para->errorNode = (UPTKGraphExecUpdateResultInfo)cuda_para->errorNode;
    UPTK_para->errorFromNode = (UPTKGraphExecUpdateResultInfo)cuda_para->errorFromNode;
    UPTK_para->result = cudaGraphExecUpdateResultToUPTKGraphExecUpdateResult(cuda_para->result);
}*/

/*const char* UPTK_symbolTocuda_symbol_v2(const char *UPTK_symbol) {
    for (size_t i = 0; i < NUM_MAPPINGS; ++i) {
        if (strcmp(UPTK_symbol, symbolMappings_v2[i].UPTK_symbol) == 0) {
            return strdup(symbolMappings_v2[i].cuda_symbol);
        }
    }
    return strdup(UPTK_symbol);
}*/

/*const char* UPTK_symbolTocuda_symbol_v2_ptsz(const char *UPTK_symbol) {
    for (size_t i = 0; i < NUM_MAPPINGS; ++i) {
        if (strcmp(UPTK_symbol, symbolMappings_v2_ptsz[i].UPTK_symbol) == 0) {
            return strdup(symbolMappings_v2_ptsz[i].cuda_symbol);
        }
    }
    return strdup(UPTK_symbol);
}*/

#if defined(__cplusplus)
}
#endif /* __cplusplus */
