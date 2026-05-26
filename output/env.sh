#!/bin/bash

# Get UPTK_ROOT by env.sh location
UPTK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export UPTK_ROOT

# Set UPTK environment variables
export UPTK_PATH=$UPTK_ROOT
export UPTK_INCLUDE_PATH=${UPTK_PATH}/include
export UPTK_LIB_PATH=${UPTK_PATH}/lib

# Add UPTK to compiler and linker paths
export C_INCLUDE_PATH=${UPTK_INCLUDE_PATH}:${C_INCLUDE_PATH}
export CPLUS_INCLUDE_PATH=${UPTK_INCLUDE_PATH}:${CPLUS_INCLUDE_PATH}
export LIBRARY_PATH=${UPTK_LIB_PATH}:${LIBRARY_PATH}
export LD_LIBRARY_PATH=${UPTK_LIB_PATH}:${LD_LIBRARY_PATH}

# Add UPTK binaries to PATH if they exist
if [ -d "${UPTK_PATH}/bin" ]; then
    export PATH=${UPTK_PATH}/bin:${PATH}
fi

echo "  UPTK environment set:"
echo "  UPTK_ROOT: $UPTK_ROOT"
echo "  include path: $UPTK_INCLUDE_PATH"
echo "  library path: $UPTK_LIB_PATH"
