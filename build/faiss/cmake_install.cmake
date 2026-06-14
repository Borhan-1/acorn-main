# Install script for directory: C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Program Files (x86)/faiss")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "C:/msys64/ucrt64/bin/objdump.exe")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/build/faiss/libfaiss.a")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/AutoTune.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/Clustering.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/IVFlib.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/Index.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/Index2Layer.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/IndexAdditiveQuantizer.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/IndexBinary.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/IndexBinaryFlat.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/IndexBinaryFromFloat.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/IndexBinaryHNSW.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/IndexBinaryHash.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/IndexBinaryIVF.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/IndexFlat.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/IndexFlatCodes.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/IndexHNSW.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/IndexACORN.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/IndexIDMap.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/IndexIVF.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/IndexIVFAdditiveQuantizer.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/IndexIVFFlat.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/IndexIVFPQ.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/IndexIVFFastScan.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/IndexIVFAdditiveQuantizerFastScan.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/IndexIVFPQFastScan.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/IndexIVFPQR.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/IndexIVFSpectralHash.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/IndexLSH.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/IndexLattice.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/IndexNNDescent.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/IndexNSG.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/IndexPQ.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/IndexFastScan.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/IndexAdditiveQuantizerFastScan.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/IndexPQFastScan.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/IndexPreTransform.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/IndexRefine.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/IndexReplicas.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/IndexRowwiseMinMax.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/IndexScalarQuantizer.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/IndexShards.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/MatrixStats.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/MetaIndexes.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/MetricType.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/VectorTransform.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/clone_index.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/index_factory.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/index_io.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/impl/AdditiveQuantizer.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/impl/AuxIndexStructures.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/impl/IDSelector.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/impl/DistanceComputer.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/impl/FaissAssert.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/impl/FaissException.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/impl/HNSW.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/impl/ACORN.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/impl/LocalSearchQuantizer.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/impl/ProductAdditiveQuantizer.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/impl/LookupTableScaler.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/impl/NNDescent.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/impl/NSG.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/impl/PolysemousTraining.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/impl/ProductQuantizer-inl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/impl/ProductQuantizer.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/impl/Quantizer.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/impl/ResidualQuantizer.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/impl/ResultHandler.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/impl/ScalarQuantizer.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/impl/ThreadedIndex-inl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/impl/ThreadedIndex.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/impl/io.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/impl/io_macros.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/impl/kmeans1d.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/impl/lattice_Zn.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/impl/platform_macros.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/impl/pq4_fast_scan.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/impl" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/impl/simd_result_handlers.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/invlists" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/invlists/BlockInvertedLists.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/invlists" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/invlists/DirectMap.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/invlists" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/invlists/InvertedLists.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/invlists" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/invlists/InvertedListsIOHook.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/utils/AlignedTable.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/utils/Heap.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/utils/WorkerThread.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/utils/distances.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/utils/extra_distances-inl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/utils/extra_distances.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/utils/fp16-fp16c.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/utils/fp16-inl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/utils/fp16.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/utils/hamming-inl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/utils/hamming.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/utils/ordered_key_value.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/utils/partitioning.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/utils/quantize_lut.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/utils/random.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/utils/simdlib.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/utils/simdlib_avx2.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/utils/simdlib_emulated.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/utils/simdlib_neon.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/utils/utils.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils/distances_fused" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/utils/distances_fused/avx512.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils/distances_fused" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/utils/distances_fused/distances_fused.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/faiss/utils/distances_fused" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/faiss/utils/distances_fused/simdlib_based.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/faiss" TYPE FILE FILES
    "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/build/cmake/faiss-config.cmake"
    "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/build/cmake/faiss-config-version.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/faiss/faiss-targets.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/faiss/faiss-targets.cmake"
         "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/build/faiss/CMakeFiles/Export/5de5d61945e1ac7c8fc01cce1738f759/faiss-targets.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/faiss/faiss-targets-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/faiss/faiss-targets.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/faiss" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/build/faiss/CMakeFiles/Export/5de5d61945e1ac7c8fc01cce1738f759/faiss-targets.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/faiss" TYPE FILE FILES "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/build/faiss/CMakeFiles/Export/5de5d61945e1ac7c8fc01cce1738f759/faiss-targets-release.cmake")
  endif()
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
if(CMAKE_INSTALL_LOCAL_ONLY)
  file(WRITE "C:/Users/Muhammad Borhan/Documents/ACORN-main/ACORN-main/build/faiss/install_local_manifest.txt"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
endif()
