#include "sparse.hpp"

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCbsr2csr(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nb, const UPTKsparseMatDescr_t descrA, const cuComplex *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim, const UPTKsparseMatDescr_t descrC, cuComplex *csrSortedValC, int *csrSortedRowPtrC, int *csrSortedColIndC)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCbsr2csr((cusparseHandle_t)handle, cuda_dirA, mb, nb, (const cusparseMatDescr_t)descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, (const cusparseMatDescr_t)descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCbsric02(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nnzb, const UPTKsparseMatDescr_t descrA, cuComplex *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim, bsric02Info_t info, UPTKsparseSolvePolicy_t policy, void *pBuffer)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseSolvePolicy_t cuda_policy = UPTKsparseSolvePolicyTocusparseSolvePolicy(policy);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCbsric02((cusparseHandle_t)handle, cuda_dirA, mb, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, (bsric02Info_t)info, cuda_policy, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCbsric02_analysis(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nnzb, const UPTKsparseMatDescr_t descrA, const cuComplex *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim, bsric02Info_t info, UPTKsparseSolvePolicy_t policy, void *pInputBuffer)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseSolvePolicy_t cuda_policy = UPTKsparseSolvePolicyTocusparseSolvePolicy(policy);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCbsric02_analysis((cusparseHandle_t)handle, cuda_dirA, mb, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, (bsric02Info_t)info, cuda_policy, pInputBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCbsric02_bufferSize(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nnzb, const UPTKsparseMatDescr_t descrA, cuComplex *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim, bsric02Info_t info, int *pBufferSizeInBytes)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCbsric02_bufferSize((cusparseHandle_t)handle, cuda_dirA, mb, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, (bsric02Info_t)info, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCbsrilu02(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nnzb, const UPTKsparseMatDescr_t descrA, cuComplex *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim, bsrilu02Info_t info, UPTKsparseSolvePolicy_t policy, void *pBuffer)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseSolvePolicy_t cuda_policy = UPTKsparseSolvePolicyTocusparseSolvePolicy(policy);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCbsrilu02((cusparseHandle_t)handle, cuda_dirA, mb, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, (bsrilu02Info_t)info, cuda_policy, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCbsrilu02_analysis(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nnzb, const UPTKsparseMatDescr_t descrA, cuComplex *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim, bsrilu02Info_t info, UPTKsparseSolvePolicy_t policy, void *pBuffer)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseSolvePolicy_t cuda_policy = UPTKsparseSolvePolicyTocusparseSolvePolicy(policy);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCbsrilu02_analysis((cusparseHandle_t)handle, cuda_dirA, mb, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, (bsrilu02Info_t)info, cuda_policy, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCbsrilu02_bufferSize(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nnzb, const UPTKsparseMatDescr_t descrA, cuComplex *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim, bsrilu02Info_t info, int *pBufferSizeInBytes)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCbsrilu02_bufferSize((cusparseHandle_t)handle, cuda_dirA, mb, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, (bsrilu02Info_t)info, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCbsrilu02_numericBoost(UPTKsparseHandle_t handle, bsrilu02Info_t info, int enable_boost, double *tol, cuComplex *boost_val)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCbsrilu02_numericBoost((cusparseHandle_t)handle, (bsrilu02Info_t)info, enable_boost, tol, boost_val);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCbsrmm(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, UPTKsparseOperation_t transA, UPTKsparseOperation_t transB, int mb, int n, int kb, int nnzb, const cuComplex *alpha, const UPTKsparseMatDescr_t descrA, const cuComplex *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, const int blockSize, const cuComplex *B, const int ldb, const cuComplex *beta, cuComplex *C, int ldc)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseOperation_t cuda_transA = UPTKsparseOperationTocusparseOperation(transA);
    cusparseOperation_t cuda_transB = UPTKsparseOperationTocusparseOperation(transB);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCbsrmm((cusparseHandle_t)handle, cuda_dirA, cuda_transA, cuda_transB, mb, n, kb, nnzb, alpha, (const cusparseMatDescr_t)descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockSize, B, ldb, beta, C, ldc);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCbsrmv(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, UPTKsparseOperation_t transA, int mb, int nb, int nnzb, const cuComplex *alpha, const UPTKsparseMatDescr_t descrA, const cuComplex *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim, const cuComplex *x, const cuComplex *beta, cuComplex *y)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseOperation_t cuda_transA = UPTKsparseOperationTocusparseOperation(transA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCbsrmv((cusparseHandle_t)handle, cuda_dirA, cuda_transA, mb, nb, nnzb, alpha, (const cusparseMatDescr_t)descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, x, beta, y);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCbsrsm2_analysis(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, UPTKsparseOperation_t transA, UPTKsparseOperation_t transXY, int mb, int n, int nnzb, const UPTKsparseMatDescr_t descrA, const cuComplex *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info, UPTKsparseSolvePolicy_t policy, void *pBuffer)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseOperation_t cuda_transA = UPTKsparseOperationTocusparseOperation(transA);
    cusparseOperation_t cuda_transX = UPTKsparseOperationTocusparseOperation(transXY);
    cusparseSolvePolicy_t cuda_policy = UPTKsparseSolvePolicyTocusparseSolvePolicy(policy);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCbsrsm2_analysis((cusparseHandle_t)handle, cuda_dirA, cuda_transA, cuda_transX, mb, n, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, (bsrsm2Info_t)info, cuda_policy, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCbsrsm2_bufferSize(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, UPTKsparseOperation_t transA, UPTKsparseOperation_t transXY, int mb, int n, int nnzb, const UPTKsparseMatDescr_t descrA, cuComplex *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info, int *pBufferSizeInBytes)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseOperation_t cuda_transA = UPTKsparseOperationTocusparseOperation(transA);
    cusparseOperation_t cuda_transX = UPTKsparseOperationTocusparseOperation(transXY);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCbsrsm2_bufferSize((cusparseHandle_t)handle, cuda_dirA, cuda_transA, cuda_transX, mb, n, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, (bsrsm2Info_t)info, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCbsrsm2_solve(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, UPTKsparseOperation_t transA, UPTKsparseOperation_t transXY, int mb, int n, int nnzb, const cuComplex *alpha, const UPTKsparseMatDescr_t descrA, const cuComplex *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info, const cuComplex *B, int ldb, cuComplex *X, int ldx, UPTKsparseSolvePolicy_t policy, void *pBuffer)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseOperation_t cuda_transA = UPTKsparseOperationTocusparseOperation(transA);
    cusparseOperation_t cuda_transX = UPTKsparseOperationTocusparseOperation(transXY);
    cusparseSolvePolicy_t cuda_policy = UPTKsparseSolvePolicyTocusparseSolvePolicy(policy);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCbsrsm2_solve((cusparseHandle_t)handle, cuda_dirA, cuda_transA, cuda_transX, mb, n, nnzb, alpha, (const cusparseMatDescr_t)descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, (bsrsm2Info_t)info, B, ldb, X, ldx, cuda_policy, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCbsrsv2_analysis(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, UPTKsparseOperation_t transA, int mb, int nnzb, const UPTKsparseMatDescr_t descrA, const cuComplex *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim, bsrsv2Info_t info, UPTKsparseSolvePolicy_t policy, void *pBuffer)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseOperation_t cuda_transA = UPTKsparseOperationTocusparseOperation(transA);
    cusparseSolvePolicy_t cuda_policy = UPTKsparseSolvePolicyTocusparseSolvePolicy(policy);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCbsrsv2_analysis((cusparseHandle_t)handle, cuda_dirA, cuda_transA, mb, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, (bsrsv2Info_t)info, cuda_policy, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCbsrsv2_bufferSize(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, UPTKsparseOperation_t transA, int mb, int nnzb, const UPTKsparseMatDescr_t descrA, cuComplex *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim, bsrsv2Info_t info, int *pBufferSizeInBytes)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseOperation_t cuda_transA = UPTKsparseOperationTocusparseOperation(transA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCbsrsv2_bufferSize((cusparseHandle_t)handle, cuda_dirA, cuda_transA, mb, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, (bsrsv2Info_t)info, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCbsrsv2_bufferSizeExt(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, UPTKsparseOperation_t transA, int mb, int nnzb, const UPTKsparseMatDescr_t descrA, cuComplex *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockSize, bsrsv2Info_t info, size_t *pBufferSize)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseOperation_t cuda_transA = UPTKsparseOperationTocusparseOperation(transA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCbsrsv2_bufferSizeExt((cusparseHandle_t)handle, cuda_dirA, cuda_transA, mb, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockSize, (bsrsv2Info_t)info, pBufferSize);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCbsrsv2_solve(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, UPTKsparseOperation_t transA, int mb, int nnzb, const cuComplex *alpha, const UPTKsparseMatDescr_t descrA, const cuComplex *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim, bsrsv2Info_t info, const cuComplex *f, cuComplex *x, UPTKsparseSolvePolicy_t policy, void *pBuffer)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseOperation_t cuda_transA = UPTKsparseOperationTocusparseOperation(transA);
    cusparseSolvePolicy_t cuda_policy = UPTKsparseSolvePolicyTocusparseSolvePolicy(policy);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCbsrsv2_solve((cusparseHandle_t)handle, cuda_dirA, cuda_transA, mb, nnzb, alpha, (const cusparseMatDescr_t)descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, (bsrsv2Info_t)info, f, x, cuda_policy, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCbsrxmv(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, UPTKsparseOperation_t transA, int sizeOfMask, int mb, int nb, int nnzb, const cuComplex *alpha, const UPTKsparseMatDescr_t descrA, const cuComplex *bsrSortedValA, const int *bsrSortedMaskPtrA, const int *bsrSortedRowPtrA, const int *bsrSortedEndPtrA, const int *bsrSortedColIndA, int blockDim, const cuComplex *x, const cuComplex *beta, cuComplex *y)
{
    cusparseDirection_t cuda_dir = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseOperation_t cuda_trans = UPTKsparseOperationTocusparseOperation(transA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCbsrxmv((cusparseHandle_t)handle, cuda_dir, cuda_trans, sizeOfMask, mb, nb, nnzb, alpha, (const cusparseMatDescr_t)descrA, bsrSortedValA, bsrSortedMaskPtrA, bsrSortedRowPtrA, bsrSortedEndPtrA, bsrSortedColIndA, blockDim, x, beta, y);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCcsr2bsr(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int m, int n, const UPTKsparseMatDescr_t descrA, const cuComplex *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, int blockDim, const UPTKsparseMatDescr_t descrC, cuComplex *bsrSortedValC, int *bsrSortedRowPtrC, int *bsrSortedColIndC)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCcsr2bsr((cusparseHandle_t)handle, cuda_dirA, m, n, (const cusparseMatDescr_t)descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, blockDim, (const cusparseMatDescr_t)descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCcsr2csr_compress(UPTKsparseHandle_t handle, int m, int n, const UPTKsparseMatDescr_t descrA, const cuComplex *csrSortedValA, const int *csrSortedColIndA, const int *csrSortedRowPtrA, int nnzA, const int *nnzPerRow, cuComplex *csrSortedValC, int *csrSortedColIndC, int *csrSortedRowPtrC, cuComplex tol)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCcsr2csr_compress((cusparseHandle_t)handle, m, n, (const cusparseMatDescr_t)descrA, csrSortedValA, csrSortedColIndA, csrSortedRowPtrA, nnzA, nnzPerRow, csrSortedValC, csrSortedColIndC, csrSortedRowPtrC, tol);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCcsr2csru(UPTKsparseHandle_t handle, int m, int n, int nnz, const UPTKsparseMatDescr_t descrA, cuComplex *csrVal, const int *csrRowPtr, int *csrColInd, csru2csrInfo_t info, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCcsr2csru((cusparseHandle_t)handle, m, n, nnz, (const cusparseMatDescr_t)descrA, csrVal, csrRowPtr, csrColInd, (csru2csrInfo_t)info, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCcsr2gebsr(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int m, int n, const UPTKsparseMatDescr_t descrA, const cuComplex *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, const UPTKsparseMatDescr_t descrC, cuComplex *bsrSortedValC, int *bsrSortedRowPtrC, int *bsrSortedColIndC, int rowBlockDim, int colBlockDim, void *pBuffer)
{
    cusparseDirection_t cuda_dir = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCcsr2gebsr((cusparseHandle_t)handle, cuda_dir, m, n, (const cusparseMatDescr_t)descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, (const cusparseMatDescr_t)descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC, rowBlockDim, colBlockDim, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCcsr2gebsr_bufferSize(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int m, int n, const UPTKsparseMatDescr_t descrA, const cuComplex *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, int rowBlockDim, int colBlockDim, int *pBufferSizeInBytes)
{
    cusparseDirection_t cuda_dir = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCcsr2gebsr_bufferSize((cusparseHandle_t)handle, cuda_dir, m, n, (const cusparseMatDescr_t)descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, rowBlockDim, colBlockDim, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCcsrcolor(UPTKsparseHandle_t handle, int m, int nnz, const UPTKsparseMatDescr_t descrA, const cuComplex *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, const float *fractionToColor, int *ncolors, int *coloring, int *reordering, const UPTKsparseColorInfo_t info)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCcsrcolor((cusparseHandle_t)handle, m, nnz, (const cusparseMatDescr_t)descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, fractionToColor, ncolors, coloring, reordering, (cusparseColorInfo_t)info);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCcsrgeam2(UPTKsparseHandle_t handle, int m, int n, const cuComplex *alpha, const UPTKsparseMatDescr_t descrA, int nnzA, const cuComplex *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, const cuComplex *beta, const UPTKsparseMatDescr_t descrB, int nnzB, const cuComplex *csrSortedValB, const int *csrSortedRowPtrB, const int *csrSortedColIndB, const UPTKsparseMatDescr_t descrC, cuComplex *csrSortedValC, int *csrSortedRowPtrC, int *csrSortedColIndC, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCcsrgeam2((cusparseHandle_t)handle, m, n, alpha, (const cusparseMatDescr_t)descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, (const cusparseMatDescr_t)descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, (const cusparseMatDescr_t)descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCcsrgeam2_bufferSizeExt(UPTKsparseHandle_t handle, int m, int n, const cuComplex *alpha, const UPTKsparseMatDescr_t descrA, int nnzA, const cuComplex *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, const cuComplex *beta, const UPTKsparseMatDescr_t descrB, int nnzB, const cuComplex *csrSortedValB, const int *csrSortedRowPtrB, const int *csrSortedColIndB, const UPTKsparseMatDescr_t descrC, const cuComplex *csrSortedValC, const int *csrSortedRowPtrC, const int *csrSortedColIndC, size_t *pBufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCcsrgeam2_bufferSizeExt((cusparseHandle_t)handle, m, n, alpha, (const cusparseMatDescr_t)descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, (const cusparseMatDescr_t)descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, (const cusparseMatDescr_t)descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCcsric02(UPTKsparseHandle_t handle, int m, int nnz, const UPTKsparseMatDescr_t descrA, cuComplex *csrSortedValA_valM, const int *csrSortedRowPtrA, const int *csrSortedColIndA, csric02Info_t info, UPTKsparseSolvePolicy_t policy, void *pBuffer)
{
    cusparseSolvePolicy_t cuda_policy = UPTKsparseSolvePolicyTocusparseSolvePolicy(policy);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCcsric02((cusparseHandle_t)handle, m, nnz, (const cusparseMatDescr_t)descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, (csric02Info_t)info, cuda_policy, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCcsric02_analysis(UPTKsparseHandle_t handle, int m, int nnz, const UPTKsparseMatDescr_t descrA, const cuComplex *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, csric02Info_t info, UPTKsparseSolvePolicy_t policy, void *pBuffer)
{
    cusparseSolvePolicy_t cuda_policy = UPTKsparseSolvePolicyTocusparseSolvePolicy(policy);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCcsric02_analysis((cusparseHandle_t)handle, m, nnz, (const cusparseMatDescr_t)descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, (csric02Info_t)info, cuda_policy, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCcsric02_bufferSize(UPTKsparseHandle_t handle, int m, int nnz, const UPTKsparseMatDescr_t descrA, cuComplex *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, csric02Info_t info, int *pBufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCcsric02_bufferSize((cusparseHandle_t)handle, m, nnz, (const cusparseMatDescr_t)descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, (csric02Info_t)info, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCcsric02_bufferSizeExt(UPTKsparseHandle_t handle, int m, int nnz, const UPTKsparseMatDescr_t descrA, cuComplex *csrSortedVal, const int *csrSortedRowPtr, const int *csrSortedColInd, csric02Info_t info, size_t *pBufferSize)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCcsric02_bufferSizeExt((cusparseHandle_t)handle, m, nnz, (const cusparseMatDescr_t)descrA, csrSortedVal, csrSortedRowPtr, csrSortedColInd, (csric02Info_t)info, pBufferSize);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCcsrilu02(UPTKsparseHandle_t handle, int m, int nnz, const UPTKsparseMatDescr_t descrA, cuComplex *csrSortedValA_valM, const int *csrSortedRowPtrA, const int *csrSortedColIndA, csrilu02Info_t info, UPTKsparseSolvePolicy_t policy, void *pBuffer)
{
    cusparseSolvePolicy_t cuda_policy = UPTKsparseSolvePolicyTocusparseSolvePolicy(policy);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCcsrilu02((cusparseHandle_t)handle, m, nnz, (const cusparseMatDescr_t)descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, (csrilu02Info_t)info, cuda_policy, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCcsrilu02_analysis(UPTKsparseHandle_t handle, int m, int nnz, const UPTKsparseMatDescr_t descrA, const cuComplex *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, csrilu02Info_t info, UPTKsparseSolvePolicy_t policy, void *pBuffer)
{
    cusparseSolvePolicy_t cuda_policy = UPTKsparseSolvePolicyTocusparseSolvePolicy(policy);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCcsrilu02_analysis((cusparseHandle_t)handle, m, nnz, (const cusparseMatDescr_t)descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, (csrilu02Info_t)info, cuda_policy, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCcsrilu02_bufferSize(UPTKsparseHandle_t handle, int m, int nnz, const UPTKsparseMatDescr_t descrA, cuComplex *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, csrilu02Info_t info, int *pBufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCcsrilu02_bufferSize((cusparseHandle_t)handle, m, nnz, (const cusparseMatDescr_t)descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, (csrilu02Info_t)info, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCcsrilu02_bufferSizeExt(UPTKsparseHandle_t handle, int m, int nnz, const UPTKsparseMatDescr_t descrA, cuComplex *csrSortedVal, const int *csrSortedRowPtr, const int *csrSortedColInd, csrilu02Info_t info, size_t *pBufferSize)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCcsrilu02_bufferSizeExt((cusparseHandle_t)handle, m, nnz, (const cusparseMatDescr_t)descrA, csrSortedVal, csrSortedRowPtr, csrSortedColInd, (csrilu02Info_t)info, pBufferSize);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCcsrilu02_numericBoost(UPTKsparseHandle_t handle, csrilu02Info_t info, int enable_boost, double *tol, cuComplex *boost_val)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCcsrilu02_numericBoost((cusparseHandle_t)handle, (csrilu02Info_t)info, enable_boost, tol, boost_val);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCcsru2csr(UPTKsparseHandle_t handle, int m, int n, int nnz, const UPTKsparseMatDescr_t descrA, cuComplex *csrVal, const int *csrRowPtr, int *csrColInd, csru2csrInfo_t info, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCcsru2csr((cusparseHandle_t)handle, m, n, nnz, (const cusparseMatDescr_t)descrA, csrVal, csrRowPtr, csrColInd, (csru2csrInfo_t)info, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCcsru2csr_bufferSizeExt(UPTKsparseHandle_t handle, int m, int n, int nnz, cuComplex *csrVal, const int *csrRowPtr, int *csrColInd, csru2csrInfo_t info, size_t *pBufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCcsru2csr_bufferSizeExt((cusparseHandle_t)handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, (csru2csrInfo_t)info, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCgebsr2csr(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nb, const UPTKsparseMatDescr_t descrA, const cuComplex *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDim, int colBlockDim, const UPTKsparseMatDescr_t descrC, cuComplex *csrSortedValC, int *csrSortedRowPtrC, int *csrSortedColIndC)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCgebsr2csr((cusparseHandle_t)handle, cuda_dirA, mb, nb, (const cusparseMatDescr_t)descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDim, colBlockDim, (const cusparseMatDescr_t)descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCgebsr2gebsc(UPTKsparseHandle_t handle, int mb, int nb, int nnzb, const cuComplex *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int rowBlockDim, int colBlockDim, cuComplex *bscVal, int *bscRowInd, int *bscColPtr, UPTKsparseAction_t copyValues, UPTKsparseIndexBase_t idxBase, void *pBuffer)
{
    cusparseAction_t cuda_copy_values = UPTKsparseActionTocusparseAction(copyValues);
    cusparseIndexBase_t cuda_idx_base = UPTKsparseIndexBaseTocusparseIndexBase(idxBase);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCgebsr2gebsc((cusparseHandle_t)handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim, bscVal, bscRowInd, bscColPtr, cuda_copy_values, cuda_idx_base, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCgebsr2gebsc_bufferSize(UPTKsparseHandle_t handle, int mb, int nb, int nnzb, const cuComplex *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int rowBlockDim, int colBlockDim, int *pBufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCgebsr2gebsc_bufferSize((cusparseHandle_t)handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCgebsr2gebsr(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nb, int nnzb, const UPTKsparseMatDescr_t descrA, const cuComplex *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, const UPTKsparseMatDescr_t descrC, cuComplex *bsrSortedValC, int *bsrSortedRowPtrC, int *bsrSortedColIndC, int rowBlockDimC, int colBlockDimC, void *pBuffer)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCgebsr2gebsr((cusparseHandle_t)handle, cuda_dirA, mb, nb, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA, colBlockDimA, (const cusparseMatDescr_t)descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC, rowBlockDimC, colBlockDimC, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCgebsr2gebsr_bufferSize(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nb, int nnzb, const UPTKsparseMatDescr_t descrA, const cuComplex *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, int rowBlockDimC, int colBlockDimC, int *pBufferSizeInBytes)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCgebsr2gebsr_bufferSize((cusparseHandle_t)handle, cuda_dirA, mb, nb, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCgemvi(UPTKsparseHandle_t handle, UPTKsparseOperation_t transA, int m, int n, const cuComplex *alpha, const cuComplex *A, int lda, int nnz, const cuComplex *xVal, const int *xInd, const cuComplex *beta, cuComplex *y, UPTKsparseIndexBase_t idxBase, void *pBuffer)
{
    cusparseOperation_t cuda_transA = UPTKsparseOperationTocusparseOperation(transA);
    cusparseIndexBase_t cuda_idxBase = UPTKsparseIndexBaseTocusparseIndexBase(idxBase);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCgemvi((cusparseHandle_t)handle, cuda_transA, m, n, alpha, A, lda, nnz, xVal, xInd, beta, y, cuda_idxBase, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCgemvi_bufferSize(UPTKsparseHandle_t handle, UPTKsparseOperation_t transA, int m, int n, int nnz, int *pBufferSize)
{
    cusparseOperation_t cuda_transA = UPTKsparseOperationTocusparseOperation(transA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCgemvi_bufferSize((cusparseHandle_t)handle, cuda_transA, m, n, nnz, pBufferSize);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCgpsvInterleavedBatch(UPTKsparseHandle_t handle, int algo, int m, cuComplex *ds, cuComplex *dl, cuComplex *d, cuComplex *du, cuComplex *dw, cuComplex *x, int batchCount, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCgpsvInterleavedBatch((cusparseHandle_t)handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCgpsvInterleavedBatch_bufferSizeExt(UPTKsparseHandle_t handle, int algo, int m, const cuComplex *ds, const cuComplex *dl, const cuComplex *d, const cuComplex *du, const cuComplex *dw, const cuComplex *x, int batchCount, size_t *pBufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCgpsvInterleavedBatch_bufferSizeExt((cusparseHandle_t)handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCgtsv2(UPTKsparseHandle_t handle, int m, int n, const cuComplex *dl, const cuComplex *d, const cuComplex *du, cuComplex *B, int ldb, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCgtsv2((cusparseHandle_t)handle, m, n, dl, d, du, B, ldb, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCgtsv2StridedBatch(UPTKsparseHandle_t handle, int m, const cuComplex *dl, const cuComplex *d, const cuComplex *du, cuComplex *x, int batchCount, int batchStride, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCgtsv2StridedBatch((cusparseHandle_t)handle, m, dl, d, du, x, batchCount, batchStride, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCgtsv2StridedBatch_bufferSizeExt(UPTKsparseHandle_t handle, int m, const cuComplex *dl, const cuComplex *d, const cuComplex *du, const cuComplex *x, int batchCount, int batchStride, size_t *bufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCgtsv2StridedBatch_bufferSizeExt((cusparseHandle_t)handle, m, dl, d, du, x, batchCount, batchStride, bufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCgtsv2_bufferSizeExt(UPTKsparseHandle_t handle, int m, int n, const cuComplex *dl, const cuComplex *d, const cuComplex *du, const cuComplex *B, int ldb, size_t *bufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCgtsv2_bufferSizeExt((cusparseHandle_t)handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCgtsv2_nopivot(UPTKsparseHandle_t handle, int m, int n, const cuComplex *dl, const cuComplex *d, const cuComplex *du, cuComplex *B, int ldb, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCgtsv2_nopivot((cusparseHandle_t)handle, m, n, dl, d, du, B, ldb, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCgtsv2_nopivot_bufferSizeExt(UPTKsparseHandle_t handle, int m, int n, const cuComplex *dl, const cuComplex *d, const cuComplex *du, const cuComplex *B, int ldb, size_t *bufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCgtsv2_nopivot_bufferSizeExt((cusparseHandle_t)handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCgtsvInterleavedBatch(UPTKsparseHandle_t handle, int algo, int m, cuComplex *dl, cuComplex *d, cuComplex *du, cuComplex *x, int batchCount, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCgtsvInterleavedBatch((cusparseHandle_t)handle, algo, m, dl, d, du, x, batchCount, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCgtsvInterleavedBatch_bufferSizeExt(UPTKsparseHandle_t handle, int algo, int m, const cuComplex *dl, const cuComplex *d, const cuComplex *du, const cuComplex *x, int batchCount, size_t *pBufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCgtsvInterleavedBatch_bufferSizeExt((cusparseHandle_t)handle, algo, m, dl, d, du, x, batchCount, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCnnz(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int m, int n, const UPTKsparseMatDescr_t descrA, const cuComplex *A, int lda, int *nnzPerRowCol, int *nnzTotalDevHostPtr)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCnnz((cusparseHandle_t)handle, cuda_dirA, m, n, (const cusparseMatDescr_t)descrA, A, lda, nnzPerRowCol, nnzTotalDevHostPtr);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCnnz_compress(UPTKsparseHandle_t handle, int m, const UPTKsparseMatDescr_t descr, const cuComplex *csrSortedValA, const int *csrSortedRowPtrA, int *nnzPerRow, int *nnzC, cuComplex tol)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCnnz_compress((cusparseHandle_t)handle, m, (const cusparseMatDescr_t)descr, csrSortedValA, csrSortedRowPtrA, nnzPerRow, nnzC, tol);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCooGet(const UPTKsparseSpMatDescr_t spMatDescr, int64_t *rows, int64_t *cols, int64_t *nnz, void **cooRowInd, void **cooColInd, void **cooValues, UPTKsparseIndexType_t *idxType, UPTKsparseIndexBase_t *idxBase, UPTKDataType *valueType)
{
    cusparseIndexType_t cuda_idxType;
    cusparseIndexBase_t cuda_idxBase;
    cudaDataType cuda_valueType;
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCooGet((const cusparseSpMatDescr_t)spMatDescr, rows, cols, nnz, cooRowInd, cooColInd, cooValues, &cuda_idxType, &cuda_idxBase, &cuda_valueType);
    *idxType = cusparseIndexTypeToUPTKsparseIndexType(cuda_idxType);
    *idxBase = cusparseIndexBaseToUPTKsparseIndexBase(cuda_idxBase);
    *valueType = cudaDataTypeToUPTKDataType(cuda_valueType);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCreate(UPTKsparseHandle_t *handle)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCreate((cusparseHandle_t *)handle);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCreateBsric02Info(bsric02Info_t *info)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCreateBsric02Info((bsric02Info_t *)info);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCreateBsrilu02Info(bsrilu02Info_t *info)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCreateBsrilu02Info((bsrilu02Info_t *)info);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCreateBsrsm2Info(bsrsm2Info_t *info)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCreateBsrsm2Info((bsrsm2Info_t *)info);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCreateBsrsv2Info(bsrsv2Info_t *info)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCreateBsrsv2Info((bsrsv2Info_t *)info);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCreateColorInfo(UPTKsparseColorInfo_t *info)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCreateColorInfo((cusparseColorInfo_t *)info);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCreateCoo(UPTKsparseSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols, int64_t nnz, void *cooRowInd, void *cooColInd, void *cooValues, UPTKsparseIndexType_t cooIdxType, UPTKsparseIndexBase_t idxBase, UPTKDataType valueType)
{
    cusparseIndexType_t cuda_cooIdxType = UPTKsparseIndexTypeTocusparseIndexType(cooIdxType);
    cusparseIndexBase_t cuda_idxBase = UPTKsparseIndexBaseTocusparseIndexBase(idxBase);
    cudaDataType cuda_valueType = UPTKDataTypeTocudaDataType(valueType);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCreateCoo((cusparseSpMatDescr_t *)spMatDescr, rows, cols, nnz, cooRowInd, cooColInd, cooValues, cuda_cooIdxType, cuda_idxBase, cuda_valueType);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCreateCsr(UPTKsparseSpMatDescr_t *spMatDescr, int64_t rows, int64_t cols, int64_t nnz, void *csrRowOffsets, void *csrColInd, void *csrValues, UPTKsparseIndexType_t csrRowOffsetsType, UPTKsparseIndexType_t csrColIndType, UPTKsparseIndexBase_t idxBase, UPTKDataType valueType)
{
    cusparseIndexType_t cuda_csrRowOffsetsType = UPTKsparseIndexTypeTocusparseIndexType(csrRowOffsetsType);
    cusparseIndexType_t cuda_csrColIndType = UPTKsparseIndexTypeTocusparseIndexType(csrColIndType);
    cusparseIndexBase_t cuda_idxBase = UPTKsparseIndexBaseTocusparseIndexBase(idxBase);
    cudaDataType cuda_valueType = UPTKDataTypeTocudaDataType(valueType);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCreateCsr((cusparseSpMatDescr_t *)spMatDescr, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues, cuda_csrRowOffsetsType, cuda_csrColIndType, cuda_idxBase, cuda_valueType);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCreateCsric02Info(csric02Info_t *info)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCreateCsric02Info((csric02Info_t *)info);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCreateCsrilu02Info(csrilu02Info_t *info)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCreateCsrilu02Info((csrilu02Info_t *)info);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCreateCsru2csrInfo(csru2csrInfo_t *info)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCreateCsru2csrInfo((csru2csrInfo_t *)info);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCreateDnMat(UPTKsparseDnMatDescr_t *dnMatDescr, int64_t rows, int64_t cols, int64_t ld, void *values, UPTKDataType valueType, UPTKsparseOrder_t order)
{
    cudaDataType cuda_valueType = UPTKDataTypeTocudaDataType(valueType);
    cusparseOrder_t cuda_order = UPTKsparseOrderTocusparseOrder(order);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCreateDnMat((cusparseDnMatDescr_t *)dnMatDescr, rows, cols, ld, values, cuda_valueType, cuda_order);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCreateDnVec(UPTKsparseDnVecDescr_t *dnVecDescr, int64_t size, void *values, UPTKDataType valueType)
{
    cudaDataType cuda_valueType = UPTKDataTypeTocudaDataType(valueType);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCreateDnVec((cusparseDnVecDescr_t *)dnVecDescr, size, values, cuda_valueType);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCreateIdentityPermutation(UPTKsparseHandle_t handle, int n, int *p)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCreateIdentityPermutation((cusparseHandle_t)handle, n, p);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCreateMatDescr(UPTKsparseMatDescr_t *descrA)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCreateMatDescr((cusparseMatDescr_t *)descrA);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCreatePruneInfo(pruneInfo_t *info)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCreatePruneInfo((pruneInfo_t *)info);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCreateSpVec(UPTKsparseSpVecDescr_t *spVecDescr, int64_t size, int64_t nnz, void *indices, void *values, UPTKsparseIndexType_t idxType, UPTKsparseIndexBase_t idxBase, UPTKDataType valueType)
{
    cusparseIndexType_t cuda_idxType = UPTKsparseIndexTypeTocusparseIndexType(idxType);
    cusparseIndexBase_t cuda_idxBase = UPTKsparseIndexBaseTocusparseIndexBase(idxBase);
    cudaDataType cuda_valueType = UPTKDataTypeTocudaDataType(valueType);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCreateSpVec((cusparseSpVecDescr_t *)spVecDescr, size, nnz, indices, values, cuda_idxType, cuda_idxBase, cuda_valueType);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCsrGet(const UPTKsparseSpMatDescr_t spMatDescr, int64_t *rows, int64_t *cols, int64_t *nnz, void **csrRowOffsets, void **csrColInd, void **csrValues, UPTKsparseIndexType_t *csrRowOffsetsType, UPTKsparseIndexType_t *csrColIndType, UPTKsparseIndexBase_t *idxBase, UPTKDataType *valueType)
{
    cusparseIndexType_t cuda_csrRowOffsetsType;
    cusparseIndexType_t cuda_csrColIndType;
    cusparseIndexBase_t cuda_idxBase;
    cudaDataType cuda_valueType;
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCsrGet((const cusparseSpMatDescr_t)spMatDescr, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues, &cuda_csrRowOffsetsType, &cuda_csrColIndType, &cuda_idxBase, &cuda_valueType);
    *csrRowOffsetsType = cusparseIndexTypeToUPTKsparseIndexType(cuda_csrRowOffsetsType);
    *csrColIndType = cusparseIndexTypeToUPTKsparseIndexType(cuda_csrColIndType);
    *idxBase = cusparseIndexBaseToUPTKsparseIndexBase(cuda_idxBase);
    *valueType = cudaDataTypeToUPTKDataType(cuda_valueType);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDbsr2csr(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nb, const UPTKsparseMatDescr_t descrA, const double *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim, const UPTKsparseMatDescr_t descrC, double *csrSortedValC, int *csrSortedRowPtrC, int *csrSortedColIndC)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDbsr2csr((cusparseHandle_t)handle, cuda_dirA, mb, nb, (const cusparseMatDescr_t)descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, (const cusparseMatDescr_t)descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDbsric02(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nnzb, const UPTKsparseMatDescr_t descrA, double *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim, bsric02Info_t info, UPTKsparseSolvePolicy_t policy, void *pBuffer)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseSolvePolicy_t cuda_policy = UPTKsparseSolvePolicyTocusparseSolvePolicy(policy);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDbsric02((cusparseHandle_t)handle, cuda_dirA, mb, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, (bsric02Info_t)info, cuda_policy, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDbsric02_analysis(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nnzb, const UPTKsparseMatDescr_t descrA, const double *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim, bsric02Info_t info, UPTKsparseSolvePolicy_t policy, void *pInputBuffer)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseSolvePolicy_t cuda_policy = UPTKsparseSolvePolicyTocusparseSolvePolicy(policy);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDbsric02_analysis((cusparseHandle_t)handle, cuda_dirA, mb, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, (bsric02Info_t)info, cuda_policy, pInputBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDbsric02_bufferSize(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nnzb, const UPTKsparseMatDescr_t descrA, double *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim, bsric02Info_t info, int *pBufferSizeInBytes)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDbsric02_bufferSize((cusparseHandle_t)handle, cuda_dirA, mb, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, (bsric02Info_t)info, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDbsrilu02(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nnzb, const UPTKsparseMatDescr_t descrA, double *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim, bsrilu02Info_t info, UPTKsparseSolvePolicy_t policy, void *pBuffer)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseSolvePolicy_t cuda_policy = UPTKsparseSolvePolicyTocusparseSolvePolicy(policy);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDbsrilu02((cusparseHandle_t)handle, cuda_dirA, mb, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, (bsrilu02Info_t)info, cuda_policy, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDbsrilu02_analysis(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nnzb, const UPTKsparseMatDescr_t descrA, double *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim, bsrilu02Info_t info, UPTKsparseSolvePolicy_t policy, void *pBuffer)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseSolvePolicy_t cuda_policy = UPTKsparseSolvePolicyTocusparseSolvePolicy(policy);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDbsrilu02_analysis((cusparseHandle_t)handle, cuda_dirA, mb, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, (bsrilu02Info_t)info, cuda_policy, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDbsrilu02_bufferSize(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nnzb, const UPTKsparseMatDescr_t descrA, double *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim, bsrilu02Info_t info, int *pBufferSizeInBytes)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDbsrilu02_bufferSize((cusparseHandle_t)handle, cuda_dirA, mb, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, (bsrilu02Info_t)info, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDbsrilu02_numericBoost(UPTKsparseHandle_t handle, bsrilu02Info_t info, int enable_boost, double *tol, double *boost_val)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDbsrilu02_numericBoost((cusparseHandle_t)handle, (bsrilu02Info_t)info, enable_boost, tol, boost_val);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDbsrmm(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, UPTKsparseOperation_t transA, UPTKsparseOperation_t transB, int mb, int n, int kb, int nnzb, const double *alpha, const UPTKsparseMatDescr_t descrA, const double *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, const int blockSize, const double *B, const int ldb, const double *beta, double *C, int ldc)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseOperation_t cuda_transA = UPTKsparseOperationTocusparseOperation(transA);
    cusparseOperation_t cuda_transB = UPTKsparseOperationTocusparseOperation(transB);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDbsrmm((cusparseHandle_t)handle, cuda_dirA, cuda_transA, cuda_transB, mb, n, kb, nnzb, alpha, (const cusparseMatDescr_t)descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockSize, B, ldb, beta, C, ldc);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDbsrmv(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, UPTKsparseOperation_t transA, int mb, int nb, int nnzb, const double *alpha, const UPTKsparseMatDescr_t descrA, const double *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim, const double *x, const double *beta, double *y)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseOperation_t cuda_transA = UPTKsparseOperationTocusparseOperation(transA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDbsrmv((cusparseHandle_t)handle, cuda_dirA, cuda_transA, mb, nb, nnzb, alpha, (const cusparseMatDescr_t)descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, x, beta, y);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDbsrsm2_analysis(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, UPTKsparseOperation_t transA, UPTKsparseOperation_t transXY, int mb, int n, int nnzb, const UPTKsparseMatDescr_t descrA, const double *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info, UPTKsparseSolvePolicy_t policy, void *pBuffer)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseOperation_t cuda_transA = UPTKsparseOperationTocusparseOperation(transA);
    cusparseOperation_t cuda_transX = UPTKsparseOperationTocusparseOperation(transXY);
    cusparseSolvePolicy_t cuda_policy = UPTKsparseSolvePolicyTocusparseSolvePolicy(policy);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDbsrsm2_analysis((cusparseHandle_t)handle, cuda_dirA, cuda_transA, cuda_transX, mb, n, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, (bsrsm2Info_t)info, cuda_policy, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDbsrsm2_bufferSize(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, UPTKsparseOperation_t transA, UPTKsparseOperation_t transXY, int mb, int n, int nnzb, const UPTKsparseMatDescr_t descrA, double *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info, int *pBufferSizeInBytes)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseOperation_t cuda_transA = UPTKsparseOperationTocusparseOperation(transA);
    cusparseOperation_t cuda_transX = UPTKsparseOperationTocusparseOperation(transXY);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDbsrsm2_bufferSize((cusparseHandle_t)handle, cuda_dirA, cuda_transA, cuda_transX, mb, n, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, (bsrsm2Info_t)info, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDbsrsm2_solve(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, UPTKsparseOperation_t transA, UPTKsparseOperation_t transXY, int mb, int n, int nnzb, const double *alpha, const UPTKsparseMatDescr_t descrA, const double *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info, const double *B, int ldb, double *X, int ldx, UPTKsparseSolvePolicy_t policy, void *pBuffer)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseOperation_t cuda_transA = UPTKsparseOperationTocusparseOperation(transA);
    cusparseOperation_t cuda_transX = UPTKsparseOperationTocusparseOperation(transXY);
    cusparseSolvePolicy_t cuda_policy = UPTKsparseSolvePolicyTocusparseSolvePolicy(policy);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDbsrsm2_solve((cusparseHandle_t)handle, cuda_dirA, cuda_transA, cuda_transX, mb, n, nnzb, alpha, (const cusparseMatDescr_t)descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, (bsrsm2Info_t)info, B, ldb, X, ldx, cuda_policy, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDbsrsv2_analysis(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, UPTKsparseOperation_t transA, int mb, int nnzb, const UPTKsparseMatDescr_t descrA, const double *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim, bsrsv2Info_t info, UPTKsparseSolvePolicy_t policy, void *pBuffer)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseOperation_t cuda_transA = UPTKsparseOperationTocusparseOperation(transA);
    cusparseSolvePolicy_t cuda_policy = UPTKsparseSolvePolicyTocusparseSolvePolicy(policy);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDbsrsv2_analysis((cusparseHandle_t)handle, cuda_dirA, cuda_transA, mb, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, (bsrsv2Info_t)info, cuda_policy, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDbsrsv2_bufferSize(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, UPTKsparseOperation_t transA, int mb, int nnzb, const UPTKsparseMatDescr_t descrA, double *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim, bsrsv2Info_t info, int *pBufferSizeInBytes)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseOperation_t cuda_transA = UPTKsparseOperationTocusparseOperation(transA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDbsrsv2_bufferSize((cusparseHandle_t)handle, cuda_dirA, cuda_transA, mb, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, (bsrsv2Info_t)info, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDbsrsv2_bufferSizeExt(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, UPTKsparseOperation_t transA, int mb, int nnzb, const UPTKsparseMatDescr_t descrA, double *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockSize, bsrsv2Info_t info, size_t *pBufferSize)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseOperation_t cuda_transA = UPTKsparseOperationTocusparseOperation(transA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDbsrsv2_bufferSizeExt((cusparseHandle_t)handle, cuda_dirA, cuda_transA, mb, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockSize, (bsrsv2Info_t)info, pBufferSize);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDbsrsv2_solve(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, UPTKsparseOperation_t transA, int mb, int nnzb, const double *alpha, const UPTKsparseMatDescr_t descrA, const double *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim, bsrsv2Info_t info, const double *f, double *x, UPTKsparseSolvePolicy_t policy, void *pBuffer)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseOperation_t cuda_transA = UPTKsparseOperationTocusparseOperation(transA);
    cusparseSolvePolicy_t cuda_policy = UPTKsparseSolvePolicyTocusparseSolvePolicy(policy);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDbsrsv2_solve((cusparseHandle_t)handle, cuda_dirA, cuda_transA, mb, nnzb, alpha, (const cusparseMatDescr_t)descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, (bsrsv2Info_t)info, f, x, cuda_policy, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDbsrxmv(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, UPTKsparseOperation_t transA, int sizeOfMask, int mb, int nb, int nnzb, const double *alpha, const UPTKsparseMatDescr_t descrA, const double *bsrSortedValA, const int *bsrSortedMaskPtrA, const int *bsrSortedRowPtrA, const int *bsrSortedEndPtrA, const int *bsrSortedColIndA, int blockDim, const double *x, const double *beta, double *y)
{
    cusparseDirection_t cuda_dir = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseOperation_t cuda_trans = UPTKsparseOperationTocusparseOperation(transA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDbsrxmv((cusparseHandle_t)handle, cuda_dir, cuda_trans, sizeOfMask, mb, nb, nnzb, alpha, (const cusparseMatDescr_t)descrA, bsrSortedValA, bsrSortedMaskPtrA, bsrSortedRowPtrA, bsrSortedEndPtrA, bsrSortedColIndA, blockDim, x, beta, y);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDcsr2bsr(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int m, int n, const UPTKsparseMatDescr_t descrA, const double *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, int blockDim, const UPTKsparseMatDescr_t descrC, double *bsrSortedValC, int *bsrSortedRowPtrC, int *bsrSortedColIndC)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDcsr2bsr((cusparseHandle_t)handle, cuda_dirA, m, n, (const cusparseMatDescr_t)descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, blockDim, (const cusparseMatDescr_t)descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDcsr2csr_compress(UPTKsparseHandle_t handle, int m, int n, const UPTKsparseMatDescr_t descrA, const double *csrSortedValA, const int *csrSortedColIndA, const int *csrSortedRowPtrA, int nnzA, const int *nnzPerRow, double *csrSortedValC, int *csrSortedColIndC, int *csrSortedRowPtrC, double tol)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDcsr2csr_compress((cusparseHandle_t)handle, m, n, (const cusparseMatDescr_t)descrA, csrSortedValA, csrSortedColIndA, csrSortedRowPtrA, nnzA, nnzPerRow, csrSortedValC, csrSortedColIndC, csrSortedRowPtrC, tol);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDcsr2csru(UPTKsparseHandle_t handle, int m, int n, int nnz, const UPTKsparseMatDescr_t descrA, double *csrVal, const int *csrRowPtr, int *csrColInd, csru2csrInfo_t info, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDcsr2csru((cusparseHandle_t)handle, m, n, nnz, (const cusparseMatDescr_t)descrA, csrVal, csrRowPtr, csrColInd, (csru2csrInfo_t)info, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDcsr2gebsr(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int m, int n, const UPTKsparseMatDescr_t descrA, const double *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, const UPTKsparseMatDescr_t descrC, double *bsrSortedValC, int *bsrSortedRowPtrC, int *bsrSortedColIndC, int rowBlockDim, int colBlockDim, void *pBuffer)
{
    cusparseDirection_t cuda_dir = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDcsr2gebsr((cusparseHandle_t)handle, cuda_dir, m, n, (const cusparseMatDescr_t)descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, (const cusparseMatDescr_t)descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC, rowBlockDim, colBlockDim, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDcsr2gebsr_bufferSize(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int m, int n, const UPTKsparseMatDescr_t descrA, const double *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, int rowBlockDim, int colBlockDim, int *pBufferSizeInBytes)
{
    cusparseDirection_t cuda_dir = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDcsr2gebsr_bufferSize((cusparseHandle_t)handle, cuda_dir, m, n, (const cusparseMatDescr_t)descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, rowBlockDim, colBlockDim, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDcsrcolor(UPTKsparseHandle_t handle, int m, int nnz, const UPTKsparseMatDescr_t descrA, const double *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, const double *fractionToColor, int *ncolors, int *coloring, int *reordering, const UPTKsparseColorInfo_t info)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDcsrcolor((cusparseHandle_t)handle, m, nnz, (const cusparseMatDescr_t)descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, fractionToColor, ncolors, coloring, reordering, (cusparseColorInfo_t)info);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDcsrgeam2(UPTKsparseHandle_t handle, int m, int n, const double *alpha, const UPTKsparseMatDescr_t descrA, int nnzA, const double *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, const double *beta, const UPTKsparseMatDescr_t descrB, int nnzB, const double *csrSortedValB, const int *csrSortedRowPtrB, const int *csrSortedColIndB, const UPTKsparseMatDescr_t descrC, double *csrSortedValC, int *csrSortedRowPtrC, int *csrSortedColIndC, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDcsrgeam2((cusparseHandle_t)handle, m, n, alpha, (const cusparseMatDescr_t)descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, (const cusparseMatDescr_t)descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, (const cusparseMatDescr_t)descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDcsrgeam2_bufferSizeExt(UPTKsparseHandle_t handle, int m, int n, const double *alpha, const UPTKsparseMatDescr_t descrA, int nnzA, const double *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, const double *beta, const UPTKsparseMatDescr_t descrB, int nnzB, const double *csrSortedValB, const int *csrSortedRowPtrB, const int *csrSortedColIndB, const UPTKsparseMatDescr_t descrC, const double *csrSortedValC, const int *csrSortedRowPtrC, const int *csrSortedColIndC, size_t *pBufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDcsrgeam2_bufferSizeExt((cusparseHandle_t)handle, m, n, alpha, (const cusparseMatDescr_t)descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, (const cusparseMatDescr_t)descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, (const cusparseMatDescr_t)descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDcsric02(UPTKsparseHandle_t handle, int m, int nnz, const UPTKsparseMatDescr_t descrA, double *csrSortedValA_valM, const int *csrSortedRowPtrA, const int *csrSortedColIndA, csric02Info_t info, UPTKsparseSolvePolicy_t policy, void *pBuffer)
{
    cusparseSolvePolicy_t cuda_policy = UPTKsparseSolvePolicyTocusparseSolvePolicy(policy);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDcsric02((cusparseHandle_t)handle, m, nnz, (const cusparseMatDescr_t)descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, (csric02Info_t)info, cuda_policy, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDcsric02_analysis(UPTKsparseHandle_t handle, int m, int nnz, const UPTKsparseMatDescr_t descrA, const double *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, csric02Info_t info, UPTKsparseSolvePolicy_t policy, void *pBuffer)
{
    cusparseSolvePolicy_t cuda_policy = UPTKsparseSolvePolicyTocusparseSolvePolicy(policy);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDcsric02_analysis((cusparseHandle_t)handle, m, nnz, (const cusparseMatDescr_t)descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, (csric02Info_t)info, cuda_policy, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDcsric02_bufferSize(UPTKsparseHandle_t handle, int m, int nnz, const UPTKsparseMatDescr_t descrA, double *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, csric02Info_t info, int *pBufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDcsric02_bufferSize((cusparseHandle_t)handle, m, nnz, (const cusparseMatDescr_t)descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, (csric02Info_t)info, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDcsric02_bufferSizeExt(UPTKsparseHandle_t handle, int m, int nnz, const UPTKsparseMatDescr_t descrA, double *csrSortedVal, const int *csrSortedRowPtr, const int *csrSortedColInd, csric02Info_t info, size_t *pBufferSize)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDcsric02_bufferSizeExt((cusparseHandle_t)handle, m, nnz, (const cusparseMatDescr_t)descrA, csrSortedVal, csrSortedRowPtr, csrSortedColInd, (csric02Info_t)info, pBufferSize);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDcsrilu02(UPTKsparseHandle_t handle, int m, int nnz, const UPTKsparseMatDescr_t descrA, double *csrSortedValA_valM, const int *csrSortedRowPtrA, const int *csrSortedColIndA, csrilu02Info_t info, UPTKsparseSolvePolicy_t policy, void *pBuffer)
{
    cusparseSolvePolicy_t cuda_policy = UPTKsparseSolvePolicyTocusparseSolvePolicy(policy);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDcsrilu02((cusparseHandle_t)handle, m, nnz, (const cusparseMatDescr_t)descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, (csrilu02Info_t)info, cuda_policy, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDcsrilu02_analysis(UPTKsparseHandle_t handle, int m, int nnz, const UPTKsparseMatDescr_t descrA, const double *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, csrilu02Info_t info, UPTKsparseSolvePolicy_t policy, void *pBuffer)
{
    cusparseSolvePolicy_t cuda_policy = UPTKsparseSolvePolicyTocusparseSolvePolicy(policy);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDcsrilu02_analysis((cusparseHandle_t)handle, m, nnz, (const cusparseMatDescr_t)descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, (csrilu02Info_t)info, cuda_policy, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDcsrilu02_bufferSize(UPTKsparseHandle_t handle, int m, int nnz, const UPTKsparseMatDescr_t descrA, double *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, csrilu02Info_t info, int *pBufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDcsrilu02_bufferSize((cusparseHandle_t)handle, m, nnz, (const cusparseMatDescr_t)descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, (csrilu02Info_t)info, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDcsrilu02_bufferSizeExt(UPTKsparseHandle_t handle, int m, int nnz, const UPTKsparseMatDescr_t descrA, double *csrSortedVal, const int *csrSortedRowPtr, const int *csrSortedColInd, csrilu02Info_t info, size_t *pBufferSize)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDcsrilu02_bufferSizeExt((cusparseHandle_t)handle, m, nnz, (const cusparseMatDescr_t)descrA, csrSortedVal, csrSortedRowPtr, csrSortedColInd, (csrilu02Info_t)info, pBufferSize);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDcsrilu02_numericBoost(UPTKsparseHandle_t handle, csrilu02Info_t info, int enable_boost, double *tol, double *boost_val)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDcsrilu02_numericBoost((cusparseHandle_t)handle, (csrilu02Info_t)info, enable_boost, tol, boost_val);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDcsru2csr(UPTKsparseHandle_t handle, int m, int n, int nnz, const UPTKsparseMatDescr_t descrA, double *csrVal, const int *csrRowPtr, int *csrColInd, csru2csrInfo_t info, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDcsru2csr((cusparseHandle_t)handle, m, n, nnz, (const cusparseMatDescr_t)descrA, csrVal, csrRowPtr, csrColInd, (csru2csrInfo_t)info, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDcsru2csr_bufferSizeExt(UPTKsparseHandle_t handle, int m, int n, int nnz, double *csrVal, const int *csrRowPtr, int *csrColInd, csru2csrInfo_t info, size_t *pBufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDcsru2csr_bufferSizeExt((cusparseHandle_t)handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, (csru2csrInfo_t)info, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDestroy(UPTKsparseHandle_t handle)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDestroy((cusparseHandle_t)handle);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDestroyBsric02Info(bsric02Info_t info)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDestroyBsric02Info((bsric02Info_t)info);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDestroyBsrilu02Info(bsrilu02Info_t info)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDestroyBsrilu02Info((bsrilu02Info_t)info);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDestroyBsrsm2Info(bsrsm2Info_t info)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDestroyBsrsm2Info((bsrsm2Info_t)info);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDestroyBsrsv2Info(bsrsv2Info_t info)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDestroyBsrsv2Info((bsrsv2Info_t)info);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDestroyColorInfo(UPTKsparseColorInfo_t info)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDestroyColorInfo((cusparseColorInfo_t)info);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDestroyCsric02Info(csric02Info_t info)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDestroyCsric02Info((csric02Info_t)info);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDestroyCsrilu02Info(csrilu02Info_t info)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDestroyCsrilu02Info((csrilu02Info_t)info);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDestroyCsru2csrInfo(csru2csrInfo_t info)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDestroyCsru2csrInfo((csru2csrInfo_t)info);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDestroyDnMat(UPTKsparseDnMatDescr_t dnMatDescr)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDestroyDnMat((cusparseDnMatDescr_t)dnMatDescr);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDestroyDnVec(UPTKsparseDnVecDescr_t dnVecDescr)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDestroyDnVec((cusparseDnVecDescr_t)dnVecDescr);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDestroyMatDescr(UPTKsparseMatDescr_t descrA)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDestroyMatDescr((cusparseMatDescr_t)descrA);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDestroyPruneInfo(pruneInfo_t info)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDestroyPruneInfo((pruneInfo_t)info);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDestroySpMat(UPTKsparseSpMatDescr_t spMatDescr)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDestroySpMat((cusparseSpMatDescr_t)spMatDescr);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDestroySpVec(UPTKsparseSpVecDescr_t spVecDescr)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDestroySpVec((cusparseSpVecDescr_t)spVecDescr);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDgebsr2csr(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nb, const UPTKsparseMatDescr_t descrA, const double *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDim, int colBlockDim, const UPTKsparseMatDescr_t descrC, double *csrSortedValC, int *csrSortedRowPtrC, int *csrSortedColIndC)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDgebsr2csr((cusparseHandle_t)handle, cuda_dirA, mb, nb, (const cusparseMatDescr_t)descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDim, colBlockDim, (const cusparseMatDescr_t)descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDgebsr2gebsc(UPTKsparseHandle_t handle, int mb, int nb, int nnzb, const double *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int rowBlockDim, int colBlockDim, double *bscVal, int *bscRowInd, int *bscColPtr, UPTKsparseAction_t copyValues, UPTKsparseIndexBase_t idxBase, void *pBuffer)
{
    cusparseAction_t cuda_copy_values = UPTKsparseActionTocusparseAction(copyValues);
    cusparseIndexBase_t cuda_idx_base = UPTKsparseIndexBaseTocusparseIndexBase(idxBase);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDgebsr2gebsc((cusparseHandle_t)handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim, bscVal, bscRowInd, bscColPtr, cuda_copy_values, cuda_idx_base, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDgebsr2gebsc_bufferSize(UPTKsparseHandle_t handle, int mb, int nb, int nnzb, const double *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int rowBlockDim, int colBlockDim, int *pBufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDgebsr2gebsc_bufferSize((cusparseHandle_t)handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDgebsr2gebsr(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nb, int nnzb, const UPTKsparseMatDescr_t descrA, const double *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, const UPTKsparseMatDescr_t descrC, double *bsrSortedValC, int *bsrSortedRowPtrC, int *bsrSortedColIndC, int rowBlockDimC, int colBlockDimC, void *pBuffer)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDgebsr2gebsr((cusparseHandle_t)handle, cuda_dirA, mb, nb, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA, colBlockDimA, (const cusparseMatDescr_t)descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC, rowBlockDimC, colBlockDimC, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDgebsr2gebsr_bufferSize(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nb, int nnzb, const UPTKsparseMatDescr_t descrA, const double *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, int rowBlockDimC, int colBlockDimC, int *pBufferSizeInBytes)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDgebsr2gebsr_bufferSize((cusparseHandle_t)handle, cuda_dirA, mb, nb, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDgemvi(UPTKsparseHandle_t handle, UPTKsparseOperation_t transA, int m, int n, const double *alpha, const double *A, int lda, int nnz, const double *xVal, const int *xInd, const double *beta, double *y, UPTKsparseIndexBase_t idxBase, void *pBuffer)
{
    cusparseOperation_t cuda_transA = UPTKsparseOperationTocusparseOperation(transA);
    cusparseIndexBase_t cuda_idxBase = UPTKsparseIndexBaseTocusparseIndexBase(idxBase);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDgemvi((cusparseHandle_t)handle, cuda_transA, m, n, alpha, A, lda, nnz, xVal, xInd, beta, y, cuda_idxBase, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDgemvi_bufferSize(UPTKsparseHandle_t handle, UPTKsparseOperation_t transA, int m, int n, int nnz, int *pBufferSize)
{
    cusparseOperation_t cuda_transA = UPTKsparseOperationTocusparseOperation(transA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDgemvi_bufferSize((cusparseHandle_t)handle, cuda_transA, m, n, nnz, pBufferSize);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDgpsvInterleavedBatch(UPTKsparseHandle_t handle, int algo, int m, double *ds, double *dl, double *d, double *du, double *dw, double *x, int batchCount, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDgpsvInterleavedBatch((cusparseHandle_t)handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDgpsvInterleavedBatch_bufferSizeExt(UPTKsparseHandle_t handle, int algo, int m, const double *ds, const double *dl, const double *d, const double *du, const double *dw, const double *x, int batchCount, size_t *pBufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDgpsvInterleavedBatch_bufferSizeExt((cusparseHandle_t)handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDgtsv2(UPTKsparseHandle_t handle, int m, int n, const double *dl, const double *d, const double *du, double *B, int ldb, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDgtsv2((cusparseHandle_t)handle, m, n, dl, d, du, B, ldb, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDgtsv2StridedBatch(UPTKsparseHandle_t handle, int m, const double *dl, const double *d, const double *du, double *x, int batchCount, int batchStride, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDgtsv2StridedBatch((cusparseHandle_t)handle, m, dl, d, du, x, batchCount, batchStride, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDgtsv2StridedBatch_bufferSizeExt(UPTKsparseHandle_t handle, int m, const double *dl, const double *d, const double *du, const double *x, int batchCount, int batchStride, size_t *bufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDgtsv2StridedBatch_bufferSizeExt((cusparseHandle_t)handle, m, dl, d, du, x, batchCount, batchStride, bufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDgtsv2_bufferSizeExt(UPTKsparseHandle_t handle, int m, int n, const double *dl, const double *d, const double *du, const double *B, int ldb, size_t *bufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDgtsv2_bufferSizeExt((cusparseHandle_t)handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDgtsv2_nopivot(UPTKsparseHandle_t handle, int m, int n, const double *dl, const double *d, const double *du, double *B, int ldb, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDgtsv2_nopivot((cusparseHandle_t)handle, m, n, dl, d, du, B, ldb, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDgtsv2_nopivot_bufferSizeExt(UPTKsparseHandle_t handle, int m, int n, const double *dl, const double *d, const double *du, const double *B, int ldb, size_t *bufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDgtsv2_nopivot_bufferSizeExt((cusparseHandle_t)handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDgtsvInterleavedBatch(UPTKsparseHandle_t handle, int algo, int m, double *dl, double *d, double *du, double *x, int batchCount, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDgtsvInterleavedBatch((cusparseHandle_t)handle, algo, m, dl, d, du, x, batchCount, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDgtsvInterleavedBatch_bufferSizeExt(UPTKsparseHandle_t handle, int algo, int m, const double *dl, const double *d, const double *du, const double *x, int batchCount, size_t *pBufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDgtsvInterleavedBatch_bufferSizeExt((cusparseHandle_t)handle, algo, m, dl, d, du, x, batchCount, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDnMatGet(const UPTKsparseDnMatDescr_t dnMatDescr, int64_t *rows, int64_t *cols, int64_t *ld, void **values, UPTKDataType *type, UPTKsparseOrder_t *order)
{
    cudaDataType cuda_valueType;
    cusparseOrder_t cuda_order;
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDnMatGet((const cusparseDnMatDescr_t)dnMatDescr, rows, cols, ld, values, &cuda_valueType, &cuda_order);
    *type = cudaDataTypeToUPTKDataType(cuda_valueType);
    *order = cusparseOrderToUPTKsparseOrder(cuda_order);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

// UNSUPPORTED
UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDnMatGetStridedBatch(const UPTKsparseDnMatDescr_t dnMatDescr, int *batchCount, int64_t *batchStride)
{
    return cusparseStatusToUPTKsparseStatus(cusparseDnMatGetStridedBatch((const cusparseDnMatDescr_t)dnMatDescr, batchCount, batchStride));
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDnMatGetValues(const UPTKsparseDnMatDescr_t dnMatDescr, void **values)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDnMatGetValues((const cusparseDnMatDescr_t)dnMatDescr, values);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

// UNSUPPORTED
UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDnMatSetStridedBatch(UPTKsparseDnMatDescr_t dnMatDescr, int batchCount, int64_t batchStride)
{
    return cusparseStatusToUPTKsparseStatus(cusparseDnMatSetStridedBatch((cusparseDnMatDescr_t)dnMatDescr, batchCount, batchStride));
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDnMatSetValues(UPTKsparseDnMatDescr_t dnMatDescr, void *values)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDnMatSetValues((cusparseDnMatDescr_t)dnMatDescr, values);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDnVecGet(const UPTKsparseDnVecDescr_t dnVecDescr, int64_t *size, void **values, UPTKDataType *valueType)
{
    cudaDataType cuda_valueType;
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDnVecGet((const cusparseDnVecDescr_t)dnVecDescr, size, values, &cuda_valueType);
    *valueType = cudaDataTypeToUPTKDataType(cuda_valueType);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDnVecGetValues(const UPTKsparseDnVecDescr_t dnVecDescr, void **values)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDnVecGetValues((const cusparseDnVecDescr_t)dnVecDescr, values);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDnVecSetValues(UPTKsparseDnVecDescr_t dnVecDescr, void *values)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDnVecSetValues((cusparseDnVecDescr_t)dnVecDescr, values);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDnnz(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int m, int n, const UPTKsparseMatDescr_t descrA, const double *A, int lda, int *nnzPerRowCol, int *nnzTotalDevHostPtr)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDnnz((cusparseHandle_t)handle, cuda_dirA, m, n, (const cusparseMatDescr_t)descrA, A, lda, nnzPerRowCol, nnzTotalDevHostPtr);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDnnz_compress(UPTKsparseHandle_t handle, int m, const UPTKsparseMatDescr_t descr, const double *csrSortedValA, const int *csrSortedRowPtrA, int *nnzPerRow, int *nnzC, double tol)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDnnz_compress((cusparseHandle_t)handle, m, (const cusparseMatDescr_t)descr, csrSortedValA, csrSortedRowPtrA, nnzPerRow, nnzC, tol);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDpruneCsr2csr(UPTKsparseHandle_t handle, int m, int n, int nnzA, const UPTKsparseMatDescr_t descrA, const double *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, const double *threshold, const UPTKsparseMatDescr_t descrC, double *csrSortedValC, const int *csrSortedRowPtrC, int *csrSortedColIndC, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDpruneCsr2csr((cusparseHandle_t)handle, m, n, nnzA, (const cusparseMatDescr_t)descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, threshold, (const cusparseMatDescr_t)descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDpruneCsr2csrByPercentage(UPTKsparseHandle_t handle, int m, int n, int nnzA, const UPTKsparseMatDescr_t descrA, const double *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage, const UPTKsparseMatDescr_t descrC, double *csrSortedValC, const int *csrSortedRowPtrC, int *csrSortedColIndC, pruneInfo_t info, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDpruneCsr2csrByPercentage((cusparseHandle_t)handle, m, n, nnzA, (const cusparseMatDescr_t)descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, (double)percentage, (const cusparseMatDescr_t)descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, (pruneInfo_t)info, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDpruneCsr2csrByPercentage_bufferSizeExt(UPTKsparseHandle_t handle, int m, int n, int nnzA, const UPTKsparseMatDescr_t descrA, const double *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage, const UPTKsparseMatDescr_t descrC, const double *csrSortedValC, const int *csrSortedRowPtrC, const int *csrSortedColIndC, pruneInfo_t info, size_t *pBufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDpruneCsr2csrByPercentage_bufferSizeExt((cusparseHandle_t)handle, m, n, nnzA, (const cusparseMatDescr_t)descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, (double)percentage, (const cusparseMatDescr_t)descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, (pruneInfo_t)info, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDpruneCsr2csrNnz(UPTKsparseHandle_t handle, int m, int n, int nnzA, const UPTKsparseMatDescr_t descrA, const double *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, const double *threshold, const UPTKsparseMatDescr_t descrC, int *csrSortedRowPtrC, int *nnzTotalDevHostPtr, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDpruneCsr2csrNnz((cusparseHandle_t)handle, m, n, nnzA, (const cusparseMatDescr_t)descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, threshold, (const cusparseMatDescr_t)descrC, csrSortedRowPtrC, nnzTotalDevHostPtr, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDpruneCsr2csrNnzByPercentage(UPTKsparseHandle_t handle, int m, int n, int nnzA, const UPTKsparseMatDescr_t descrA, const double *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage, const UPTKsparseMatDescr_t descrC, int *csrSortedRowPtrC, int *nnzTotalDevHostPtr, pruneInfo_t info, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDpruneCsr2csrNnzByPercentage((cusparseHandle_t)handle, m, n, nnzA, (const cusparseMatDescr_t)descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, (double)percentage, (const cusparseMatDescr_t)descrC, csrSortedRowPtrC, nnzTotalDevHostPtr, (pruneInfo_t)info, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDpruneCsr2csr_bufferSizeExt(UPTKsparseHandle_t handle, int m, int n, int nnzA, const UPTKsparseMatDescr_t descrA, const double *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, const double *threshold, const UPTKsparseMatDescr_t descrC, const double *csrSortedValC, const int *csrSortedRowPtrC, const int *csrSortedColIndC, size_t *pBufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDpruneCsr2csr_bufferSizeExt((cusparseHandle_t)handle, m, n, nnzA, (const cusparseMatDescr_t)descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, threshold, (const cusparseMatDescr_t)descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDpruneDense2csr(UPTKsparseHandle_t handle, int m, int n, const double *A, int lda, const double *threshold, const UPTKsparseMatDescr_t descrC, double *csrSortedValC, const int *csrSortedRowPtrC, int *csrSortedColIndC, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDpruneDense2csr((cusparseHandle_t)handle, m, n, A, lda, threshold, (const cusparseMatDescr_t)descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDpruneDense2csrByPercentage(UPTKsparseHandle_t handle, int m, int n, const double *A, int lda, float percentage, const UPTKsparseMatDescr_t descrC, double *csrSortedValC, const int *csrSortedRowPtrC, int *csrSortedColIndC, pruneInfo_t info, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDpruneDense2csrByPercentage((cusparseHandle_t)handle, m, n, A, lda, (double)percentage, (const cusparseMatDescr_t)descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, (pruneInfo_t)info, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDpruneDense2csrByPercentage_bufferSizeExt(UPTKsparseHandle_t handle, int m, int n, const double *A, int lda, float percentage, const UPTKsparseMatDescr_t descrC, const double *csrSortedValC, const int *csrSortedRowPtrC, const int *csrSortedColIndC, pruneInfo_t info, size_t *pBufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDpruneDense2csrByPercentage_bufferSizeExt((cusparseHandle_t)handle, m, n, A, lda, (double)percentage, (const cusparseMatDescr_t)descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, (pruneInfo_t)info, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDpruneDense2csrNnz(UPTKsparseHandle_t handle, int m, int n, const double *A, int lda, const double *threshold, const UPTKsparseMatDescr_t descrC, int *csrSortedRowPtrC, int *nnzTotalDevHostPtr, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDpruneDense2csrNnz((cusparseHandle_t)handle, m, n, A, lda, threshold, (const cusparseMatDescr_t)descrC, csrSortedRowPtrC, nnzTotalDevHostPtr, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDpruneDense2csrNnzByPercentage(UPTKsparseHandle_t handle, int m, int n, const double *A, int lda, float percentage, const UPTKsparseMatDescr_t descrC, int *csrRowPtrC, int *nnzTotalDevHostPtr, pruneInfo_t info, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDpruneDense2csrNnzByPercentage((cusparseHandle_t)handle, m, n, A, lda, (double)percentage, (const cusparseMatDescr_t)descrC, csrRowPtrC, nnzTotalDevHostPtr, (pruneInfo_t)info, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDpruneDense2csr_bufferSizeExt(UPTKsparseHandle_t handle, int m, int n, const double *A, int lda, const double *threshold, const UPTKsparseMatDescr_t descrC, const double *csrSortedValC, const int *csrSortedRowPtrC, const int *csrSortedColIndC, size_t *pBufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseDpruneDense2csr_bufferSizeExt((cusparseHandle_t)handle, m, n, A, lda, threshold, (const cusparseMatDescr_t)descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

const char *UPTKSPARSEAPI UPTKsparseGetErrorName(UPTKsparseStatus_t status)
{
    switch (status)
    {
    case UPTKSPARSE_STATUS_SUCCESS:
        return "UPTKSPARSE_STATUS_SUCCESS";
    case UPTKSPARSE_STATUS_NOT_INITIALIZED:
        return "UPTKSPARSE_STATUS_NOT_INITIALIZED";
    case UPTKSPARSE_STATUS_ALLOC_FAILED:
        return "UPTKSPARSE_STATUS_ALLOC_FAILED";
    case UPTKSPARSE_STATUS_INVALID_VALUE:
        return "UPTKSPARSE_STATUS_INVALID_VALUE";
    case UPTKSPARSE_STATUS_ARCH_MISMATCH:
        return "UPTKSPARSE_STATUS_ARCH_MISMATCH";
    case UPTKSPARSE_STATUS_MAPPING_ERROR:
        return "UPTKSPARSE_STATUS_MAPPING_ERROR";
    case UPTKSPARSE_STATUS_EXECUTION_FAILED:
        return "UPTKSPARSE_STATUS_EXECUTION_FAILED";
    case UPTKSPARSE_STATUS_INTERNAL_ERROR:
        return "UPTKSPARSE_STATUS_INTERNAL_ERROR";
    case UPTKSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        return "UPTKSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    case UPTKSPARSE_STATUS_ZERO_PIVOT:
        return "UPTKSPARSE_STATUS_ZERO_PIVOT";
    case UPTKSPARSE_STATUS_NOT_SUPPORTED:
        return "UPTKSPARSE_STATUS_NOT_SUPPORTED";
    case UPTKSPARSE_STATUS_INSUFFICIENT_RESOURCES:
        return "UPTKSPARSE_STATUS_INSUFFICIENT_RESOURCES";
    default:
        break;
    }

    return "unrecognized error code";
}

const char *UPTKSPARSEAPI UPTKsparseGetErrorString(UPTKsparseStatus_t status)
{
    switch (status)
    {
    case UPTKSPARSE_STATUS_SUCCESS:
        return "success";
    case UPTKSPARSE_STATUS_NOT_INITIALIZED:
        return "initialization error";
    case UPTKSPARSE_STATUS_ALLOC_FAILED:
        return "out of memory";
    case UPTKSPARSE_STATUS_INVALID_VALUE:
        return "invalid value";
    case UPTKSPARSE_STATUS_ARCH_MISMATCH:
        return "architecture mismatch";
    case UPTKSPARSE_STATUS_MAPPING_ERROR:
        return "texture memory mapping error";
    case UPTKSPARSE_STATUS_EXECUTION_FAILED:
        return "kernel launch failure";
    case UPTKSPARSE_STATUS_INTERNAL_ERROR:
        return "internal error";
    case UPTKSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        return "matrix type not supported";
    case UPTKSPARSE_STATUS_ZERO_PIVOT:
        return "zero pivot";
    case UPTKSPARSE_STATUS_NOT_SUPPORTED:
        return "operation not supported";
    case UPTKSPARSE_STATUS_INSUFFICIENT_RESOURCES:
        return "insufficient resources";
    default:
        break;
    }

    return "unrecognized error code";
}

UPTKsparseDiagType_t UPTKSPARSEAPI UPTKsparseGetMatDiagType(const UPTKsparseMatDescr_t descrA)
{
    cusparseDiagType_t cuda_res;
    cuda_res = cusparseGetMatDiagType((const cusparseMatDescr_t)descrA);
    return cusparseDiagTypeToUPTKsparseDiagType(cuda_res);
}

UPTKsparseFillMode_t UPTKSPARSEAPI UPTKsparseGetMatFillMode(const UPTKsparseMatDescr_t descrA)
{
    cusparseFillMode_t cuda_res;
    cuda_res = cusparseGetMatFillMode((const cusparseMatDescr_t)descrA);
    return cusparseFillModeToUPTKsparseFillMode(cuda_res);
}

UPTKsparseIndexBase_t UPTKSPARSEAPI UPTKsparseGetMatIndexBase(const UPTKsparseMatDescr_t descrA)
{
    cusparseIndexBase_t cuda_res;
    cuda_res = cusparseGetMatIndexBase((const cusparseMatDescr_t)descrA);
    return cusparseIndexBaseToUPTKsparseIndexBase(cuda_res);
}

UPTKsparseMatrixType_t UPTKSPARSEAPI UPTKsparseGetMatType(const UPTKsparseMatDescr_t descrA)
{
    cusparseMatrixType_t cuda_res;
    cuda_res = cusparseGetMatType((const cusparseMatDescr_t)descrA);
    return cusparseMatrixTypeToUPTKsparseMatrixType(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseGetPointerMode(UPTKsparseHandle_t handle, UPTKsparsePointerMode_t *mode)
{
    cusparsePointerMode_t cuda_mode;
    cusparseStatus_t cuda_res;
    cuda_res = cusparseGetPointerMode((cusparseHandle_t)handle, &cuda_mode);
    *mode = cusparsePointerModeToUPTKsparsePointerMode(cuda_mode);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseGetProperty(libraryPropertyType type, int *value)
{
    if (NULL == value)
    {
        return UPTKSPARSE_STATUS_INVALID_VALUE;
    }

    switch (type)
    {
    case MAJOR_VERSION:
        *value = UPTKSPARSE_VER_MAJOR;
        break;
    case MINOR_VERSION:
        *value = UPTKSPARSE_VER_MINOR;
        break;
    case PATCH_LEVEL:
        *value = UPTKSPARSE_VER_PATCH;
        break;
    default:
        return UPTKSPARSE_STATUS_INVALID_VALUE;
    }

    return UPTKSPARSE_STATUS_SUCCESS;
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseGetStream(UPTKsparseHandle_t handle, UPTKStream_t *streamId)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseGetStream((cusparseHandle_t)handle, (cudaStream_t *)streamId);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseGetVersion(UPTKsparseHandle_t handle, int *version)
{
    if (NULL == handle)
    {
        return UPTKSPARSE_STATUS_NOT_INITIALIZED;
    }

    if (NULL == version)
    {
        return UPTKSPARSE_STATUS_INVALID_VALUE;
    }

    *version = UPTKSPARSE_VERSION;

    return UPTKSPARSE_STATUS_SUCCESS;
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSbsr2csr(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nb, const UPTKsparseMatDescr_t descrA, const float *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim, const UPTKsparseMatDescr_t descrC, float *csrSortedValC, int *csrSortedRowPtrC, int *csrSortedColIndC)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSbsr2csr((cusparseHandle_t)handle, cuda_dirA, mb, nb, (const cusparseMatDescr_t)descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, (const cusparseMatDescr_t)descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSbsric02(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nnzb, const UPTKsparseMatDescr_t descrA, float *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim, bsric02Info_t info, UPTKsparseSolvePolicy_t policy, void *pBuffer)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseSolvePolicy_t cuda_policy = UPTKsparseSolvePolicyTocusparseSolvePolicy(policy);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSbsric02((cusparseHandle_t)handle, cuda_dirA, mb, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, (bsric02Info_t)info, cuda_policy, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSbsric02_analysis(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nnzb, const UPTKsparseMatDescr_t descrA, const float *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim, bsric02Info_t info, UPTKsparseSolvePolicy_t policy, void *pInputBuffer)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseSolvePolicy_t cuda_policy = UPTKsparseSolvePolicyTocusparseSolvePolicy(policy);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSbsric02_analysis((cusparseHandle_t)handle, cuda_dirA, mb, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, (bsric02Info_t)info, cuda_policy, pInputBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSbsric02_bufferSize(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nnzb, const UPTKsparseMatDescr_t descrA, float *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim, bsric02Info_t info, int *pBufferSizeInBytes)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSbsric02_bufferSize((cusparseHandle_t)handle, cuda_dirA, mb, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, (bsric02Info_t)info, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSbsrilu02(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nnzb, const UPTKsparseMatDescr_t descrA, float *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim, bsrilu02Info_t info, UPTKsparseSolvePolicy_t policy, void *pBuffer)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseSolvePolicy_t cuda_policy = UPTKsparseSolvePolicyTocusparseSolvePolicy(policy);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSbsrilu02((cusparseHandle_t)handle, cuda_dirA, mb, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, (bsrilu02Info_t)info, cuda_policy, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSbsrilu02_analysis(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nnzb, const UPTKsparseMatDescr_t descrA, float *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim, bsrilu02Info_t info, UPTKsparseSolvePolicy_t policy, void *pBuffer)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseSolvePolicy_t cuda_policy = UPTKsparseSolvePolicyTocusparseSolvePolicy(policy);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSbsrilu02_analysis((cusparseHandle_t)handle, cuda_dirA, mb, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, (bsrilu02Info_t)info, cuda_policy, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSbsrilu02_bufferSize(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nnzb, const UPTKsparseMatDescr_t descrA, float *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim, bsrilu02Info_t info, int *pBufferSizeInBytes)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSbsrilu02_bufferSize((cusparseHandle_t)handle, cuda_dirA, mb, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, (bsrilu02Info_t)info, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSbsrilu02_numericBoost(UPTKsparseHandle_t handle, bsrilu02Info_t info, int enable_boost, double *tol, float *boost_val)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSbsrilu02_numericBoost((cusparseHandle_t)handle, (bsrilu02Info_t)info, enable_boost, tol, boost_val);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSbsrmm(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, UPTKsparseOperation_t transA, UPTKsparseOperation_t transB, int mb, int n, int kb, int nnzb, const float *alpha, const UPTKsparseMatDescr_t descrA, const float *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, const int blockSize, const float *B, const int ldb, const float *beta, float *C, int ldc)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseOperation_t cuda_transA = UPTKsparseOperationTocusparseOperation(transA);
    cusparseOperation_t cuda_transB = UPTKsparseOperationTocusparseOperation(transB);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSbsrmm((cusparseHandle_t)handle, cuda_dirA, cuda_transA, cuda_transB, mb, n, kb, nnzb, alpha, (const cusparseMatDescr_t)descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockSize, B, ldb, beta, C, ldc);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSbsrmv(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, UPTKsparseOperation_t transA, int mb, int nb, int nnzb, const float *alpha, const UPTKsparseMatDescr_t descrA, const float *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim, const float *x, const float *beta, float *y)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseOperation_t cuda_transA = UPTKsparseOperationTocusparseOperation(transA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSbsrmv((cusparseHandle_t)handle, cuda_dirA, cuda_transA, mb, nb, nnzb, alpha, (const cusparseMatDescr_t)descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, x, beta, y);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSbsrsm2_analysis(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, UPTKsparseOperation_t transA, UPTKsparseOperation_t transXY, int mb, int n, int nnzb, const UPTKsparseMatDescr_t descrA, const float *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info, UPTKsparseSolvePolicy_t policy, void *pBuffer)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseOperation_t cuda_transA = UPTKsparseOperationTocusparseOperation(transA);
    cusparseOperation_t cuda_transX = UPTKsparseOperationTocusparseOperation(transXY);
    cusparseSolvePolicy_t cuda_policy = UPTKsparseSolvePolicyTocusparseSolvePolicy(policy);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSbsrsm2_analysis((cusparseHandle_t)handle, cuda_dirA, cuda_transA, cuda_transX, mb, n, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, (bsrsm2Info_t)info, cuda_policy, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSbsrsm2_bufferSize(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, UPTKsparseOperation_t transA, UPTKsparseOperation_t transXY, int mb, int n, int nnzb, const UPTKsparseMatDescr_t descrA, float *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info, int *pBufferSizeInBytes)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseOperation_t cuda_transA = UPTKsparseOperationTocusparseOperation(transA);
    cusparseOperation_t cuda_transX = UPTKsparseOperationTocusparseOperation(transXY);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSbsrsm2_bufferSize((cusparseHandle_t)handle, cuda_dirA, cuda_transA, cuda_transX, mb, n, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, (bsrsm2Info_t)info, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSbsrsm2_solve(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, UPTKsparseOperation_t transA, UPTKsparseOperation_t transXY, int mb, int n, int nnzb, const float *alpha, const UPTKsparseMatDescr_t descrA, const float *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info, const float *B, int ldb, float *X, int ldx, UPTKsparseSolvePolicy_t policy, void *pBuffer)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseOperation_t cuda_transA = UPTKsparseOperationTocusparseOperation(transA);
    cusparseOperation_t cuda_transX = UPTKsparseOperationTocusparseOperation(transXY);
    cusparseSolvePolicy_t cuda_policy = UPTKsparseSolvePolicyTocusparseSolvePolicy(policy);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSbsrsm2_solve((cusparseHandle_t)handle, cuda_dirA, cuda_transA, cuda_transX, mb, n, nnzb, alpha, (const cusparseMatDescr_t)descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, (bsrsm2Info_t)info, B, ldb, X, ldx, cuda_policy, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSbsrsv2_analysis(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, UPTKsparseOperation_t transA, int mb, int nnzb, const UPTKsparseMatDescr_t descrA, const float *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim, bsrsv2Info_t info, UPTKsparseSolvePolicy_t policy, void *pBuffer)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseOperation_t cuda_transA = UPTKsparseOperationTocusparseOperation(transA);
    cusparseSolvePolicy_t cuda_policy = UPTKsparseSolvePolicyTocusparseSolvePolicy(policy);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSbsrsv2_analysis((cusparseHandle_t)handle, cuda_dirA, cuda_transA, mb, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, (bsrsv2Info_t)info, cuda_policy, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSbsrsv2_bufferSize(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, UPTKsparseOperation_t transA, int mb, int nnzb, const UPTKsparseMatDescr_t descrA, float *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim, bsrsv2Info_t info, int *pBufferSizeInBytes)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseOperation_t cuda_transA = UPTKsparseOperationTocusparseOperation(transA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSbsrsv2_bufferSize((cusparseHandle_t)handle, cuda_dirA, cuda_transA, mb, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, (bsrsv2Info_t)info, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSbsrsv2_bufferSizeExt(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, UPTKsparseOperation_t transA, int mb, int nnzb, const UPTKsparseMatDescr_t descrA, float *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockSize, bsrsv2Info_t info, size_t *pBufferSize)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseOperation_t cuda_transA = UPTKsparseOperationTocusparseOperation(transA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSbsrsv2_bufferSizeExt((cusparseHandle_t)handle, cuda_dirA, cuda_transA, mb, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockSize, (bsrsv2Info_t)info, pBufferSize);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSbsrsv2_solve(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, UPTKsparseOperation_t transA, int mb, int nnzb, const float *alpha, const UPTKsparseMatDescr_t descrA, const float *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim, bsrsv2Info_t info, const float *f, float *x, UPTKsparseSolvePolicy_t policy, void *pBuffer)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseOperation_t cuda_transA = UPTKsparseOperationTocusparseOperation(transA);
    cusparseSolvePolicy_t cuda_policy = UPTKsparseSolvePolicyTocusparseSolvePolicy(policy);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSbsrsv2_solve((cusparseHandle_t)handle, cuda_dirA, cuda_transA, mb, nnzb, alpha, (const cusparseMatDescr_t)descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, (bsrsv2Info_t)info, f, x, cuda_policy, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSbsrxmv(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, UPTKsparseOperation_t transA, int sizeOfMask, int mb, int nb, int nnzb, const float *alpha, const UPTKsparseMatDescr_t descrA, const float *bsrSortedValA, const int *bsrSortedMaskPtrA, const int *bsrSortedRowPtrA, const int *bsrSortedEndPtrA, const int *bsrSortedColIndA, int blockDim, const float *x, const float *beta, float *y)
{
    cusparseDirection_t cuda_dir = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseOperation_t cuda_trans = UPTKsparseOperationTocusparseOperation(transA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSbsrxmv((cusparseHandle_t)handle, cuda_dir, cuda_trans, sizeOfMask, mb, nb, nnzb, alpha, (const cusparseMatDescr_t)descrA, bsrSortedValA, bsrSortedMaskPtrA, bsrSortedRowPtrA, bsrSortedEndPtrA, bsrSortedColIndA, blockDim, x, beta, y);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseScsr2bsr(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int m, int n, const UPTKsparseMatDescr_t descrA, const float *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, int blockDim, const UPTKsparseMatDescr_t descrC, float *bsrSortedValC, int *bsrSortedRowPtrC, int *bsrSortedColIndC)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseScsr2bsr((cusparseHandle_t)handle, cuda_dirA, m, n, (const cusparseMatDescr_t)descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, blockDim, (const cusparseMatDescr_t)descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseScsr2csr_compress(UPTKsparseHandle_t handle, int m, int n, const UPTKsparseMatDescr_t descrA, const float *csrSortedValA, const int *csrSortedColIndA, const int *csrSortedRowPtrA, int nnzA, const int *nnzPerRow, float *csrSortedValC, int *csrSortedColIndC, int *csrSortedRowPtrC, float tol)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseScsr2csr_compress((cusparseHandle_t)handle, m, n, (const cusparseMatDescr_t)descrA, csrSortedValA, csrSortedColIndA, csrSortedRowPtrA, nnzA, nnzPerRow, csrSortedValC, csrSortedColIndC, csrSortedRowPtrC, tol);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseScsr2csru(UPTKsparseHandle_t handle, int m, int n, int nnz, const UPTKsparseMatDescr_t descrA, float *csrVal, const int *csrRowPtr, int *csrColInd, csru2csrInfo_t info, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseScsr2csru((cusparseHandle_t)handle, m, n, nnz, (const cusparseMatDescr_t)descrA, csrVal, csrRowPtr, csrColInd, (csru2csrInfo_t)info, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseScsr2gebsr(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int m, int n, const UPTKsparseMatDescr_t descrA, const float *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, const UPTKsparseMatDescr_t descrC, float *bsrSortedValC, int *bsrSortedRowPtrC, int *bsrSortedColIndC, int rowBlockDim, int colBlockDim, void *pBuffer)
{
    cusparseDirection_t cuda_dir = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseScsr2gebsr((cusparseHandle_t)handle, cuda_dir, m, n, (const cusparseMatDescr_t)descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, (const cusparseMatDescr_t)descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC, rowBlockDim, colBlockDim, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseScsr2gebsr_bufferSize(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int m, int n, const UPTKsparseMatDescr_t descrA, const float *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, int rowBlockDim, int colBlockDim, int *pBufferSizeInBytes)
{
    cusparseDirection_t cuda_dir = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseScsr2gebsr_bufferSize((cusparseHandle_t)handle, cuda_dir, m, n, (const cusparseMatDescr_t)descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, rowBlockDim, colBlockDim, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseScsrcolor(UPTKsparseHandle_t handle, int m, int nnz, const UPTKsparseMatDescr_t descrA, const float *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, const float *fractionToColor, int *ncolors, int *coloring, int *reordering, const UPTKsparseColorInfo_t info)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseScsrcolor((cusparseHandle_t)handle, m, nnz, (const cusparseMatDescr_t)descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, fractionToColor, ncolors, coloring, reordering, (cusparseColorInfo_t)info);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseScsrgeam2(UPTKsparseHandle_t handle, int m, int n, const float *alpha, const UPTKsparseMatDescr_t descrA, int nnzA, const float *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, const float *beta, const UPTKsparseMatDescr_t descrB, int nnzB, const float *csrSortedValB, const int *csrSortedRowPtrB, const int *csrSortedColIndB, const UPTKsparseMatDescr_t descrC, float *csrSortedValC, int *csrSortedRowPtrC, int *csrSortedColIndC, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseScsrgeam2((cusparseHandle_t)handle, m, n, alpha, (const cusparseMatDescr_t)descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, (const cusparseMatDescr_t)descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, (const cusparseMatDescr_t)descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseScsrgeam2_bufferSizeExt(UPTKsparseHandle_t handle, int m, int n, const float *alpha, const UPTKsparseMatDescr_t descrA, int nnzA, const float *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, const float *beta, const UPTKsparseMatDescr_t descrB, int nnzB, const float *csrSortedValB, const int *csrSortedRowPtrB, const int *csrSortedColIndB, const UPTKsparseMatDescr_t descrC, const float *csrSortedValC, const int *csrSortedRowPtrC, const int *csrSortedColIndC, size_t *pBufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseScsrgeam2_bufferSizeExt((cusparseHandle_t)handle, m, n, alpha, (const cusparseMatDescr_t)descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, (const cusparseMatDescr_t)descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, (const cusparseMatDescr_t)descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseScsric02(UPTKsparseHandle_t handle, int m, int nnz, const UPTKsparseMatDescr_t descrA, float *csrSortedValA_valM, const int *csrSortedRowPtrA, const int *csrSortedColIndA, csric02Info_t info, UPTKsparseSolvePolicy_t policy, void *pBuffer)
{
    cusparseSolvePolicy_t cuda_policy = UPTKsparseSolvePolicyTocusparseSolvePolicy(policy);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseScsric02((cusparseHandle_t)handle, m, nnz, (const cusparseMatDescr_t)descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, (csric02Info_t)info, cuda_policy, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseScsric02_analysis(UPTKsparseHandle_t handle, int m, int nnz, const UPTKsparseMatDescr_t descrA, const float *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, csric02Info_t info, UPTKsparseSolvePolicy_t policy, void *pBuffer)
{
    cusparseSolvePolicy_t cuda_policy = UPTKsparseSolvePolicyTocusparseSolvePolicy(policy);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseScsric02_analysis((cusparseHandle_t)handle, m, nnz, (const cusparseMatDescr_t)descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, (csric02Info_t)info, cuda_policy, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseScsric02_bufferSize(UPTKsparseHandle_t handle, int m, int nnz, const UPTKsparseMatDescr_t descrA, float *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, csric02Info_t info, int *pBufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseScsric02_bufferSize((cusparseHandle_t)handle, m, nnz, (const cusparseMatDescr_t)descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, (csric02Info_t)info, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseScsric02_bufferSizeExt(UPTKsparseHandle_t handle, int m, int nnz, const UPTKsparseMatDescr_t descrA, float *csrSortedVal, const int *csrSortedRowPtr, const int *csrSortedColInd, csric02Info_t info, size_t *pBufferSize)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseScsric02_bufferSizeExt((cusparseHandle_t)handle, m, nnz, (const cusparseMatDescr_t)descrA, csrSortedVal, csrSortedRowPtr, csrSortedColInd, (csric02Info_t)info, pBufferSize);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseScsrilu02(UPTKsparseHandle_t handle, int m, int nnz, const UPTKsparseMatDescr_t descrA, float *csrSortedValA_valM, const int *csrSortedRowPtrA, const int *csrSortedColIndA, csrilu02Info_t info, UPTKsparseSolvePolicy_t policy, void *pBuffer)
{
    cusparseSolvePolicy_t cuda_policy = UPTKsparseSolvePolicyTocusparseSolvePolicy(policy);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseScsrilu02((cusparseHandle_t)handle, m, nnz, (const cusparseMatDescr_t)descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, (csrilu02Info_t)info, cuda_policy, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseScsrilu02_analysis(UPTKsparseHandle_t handle, int m, int nnz, const UPTKsparseMatDescr_t descrA, const float *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, csrilu02Info_t info, UPTKsparseSolvePolicy_t policy, void *pBuffer)
{
    cusparseSolvePolicy_t cuda_policy = UPTKsparseSolvePolicyTocusparseSolvePolicy(policy);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseScsrilu02_analysis((cusparseHandle_t)handle, m, nnz, (const cusparseMatDescr_t)descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, (csrilu02Info_t)info, cuda_policy, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseScsrilu02_bufferSize(UPTKsparseHandle_t handle, int m, int nnz, const UPTKsparseMatDescr_t descrA, float *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, csrilu02Info_t info, int *pBufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseScsrilu02_bufferSize((cusparseHandle_t)handle, m, nnz, (const cusparseMatDescr_t)descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, (csrilu02Info_t)info, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseScsrilu02_bufferSizeExt(UPTKsparseHandle_t handle, int m, int nnz, const UPTKsparseMatDescr_t descrA, float *csrSortedVal, const int *csrSortedRowPtr, const int *csrSortedColInd, csrilu02Info_t info, size_t *pBufferSize)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseScsrilu02_bufferSizeExt((cusparseHandle_t)handle, m, nnz, (const cusparseMatDescr_t)descrA, csrSortedVal, csrSortedRowPtr, csrSortedColInd, (csrilu02Info_t)info, pBufferSize);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseScsrilu02_numericBoost(UPTKsparseHandle_t handle, csrilu02Info_t info, int enable_boost, double *tol, float *boost_val)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseScsrilu02_numericBoost((cusparseHandle_t)handle, (csrilu02Info_t)info, enable_boost, tol, boost_val);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseScsru2csr(UPTKsparseHandle_t handle, int m, int n, int nnz, const UPTKsparseMatDescr_t descrA, float *csrVal, const int *csrRowPtr, int *csrColInd, csru2csrInfo_t info, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseScsru2csr((cusparseHandle_t)handle, m, n, nnz, (const cusparseMatDescr_t)descrA, csrVal, csrRowPtr, csrColInd, (csru2csrInfo_t)info, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseScsru2csr_bufferSizeExt(UPTKsparseHandle_t handle, int m, int n, int nnz, float *csrVal, const int *csrRowPtr, int *csrColInd, csru2csrInfo_t info, size_t *pBufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseScsru2csr_bufferSizeExt((cusparseHandle_t)handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, (csru2csrInfo_t)info, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSetMatDiagType(UPTKsparseMatDescr_t descrA, UPTKsparseDiagType_t diagType)
{
    cusparseDiagType_t cuda_diagType = UPTKsparseDiagTypeTocusparseDiagType(diagType);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSetMatDiagType((cusparseMatDescr_t)descrA, cuda_diagType);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSetMatFillMode(UPTKsparseMatDescr_t descrA, UPTKsparseFillMode_t fillMode)
{
    cusparseFillMode_t cuda_fillMode = UPTKsparseFillModeTocusparseFillMode(fillMode);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSetMatFillMode((cusparseMatDescr_t)descrA, cuda_fillMode);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSetMatIndexBase(UPTKsparseMatDescr_t descrA, UPTKsparseIndexBase_t base)
{
    cusparseIndexBase_t cuda_base = UPTKsparseIndexBaseTocusparseIndexBase(base);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSetMatIndexBase((cusparseMatDescr_t)descrA, cuda_base);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSetMatType(UPTKsparseMatDescr_t descrA, UPTKsparseMatrixType_t type)
{
    cusparseMatrixType_t cuda_type = UPTKsparseMatrixTypeTocusparseMatrixType(type);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSetMatType((cusparseMatDescr_t)descrA, cuda_type);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSetPointerMode(UPTKsparseHandle_t handle, UPTKsparsePointerMode_t mode)
{
    cusparsePointerMode_t cuda_mode = UPTKsparsePointerModeTocusparsePointerMode(mode);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSetPointerMode((cusparseHandle_t)handle, cuda_mode);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSetStream(UPTKsparseHandle_t handle, UPTKStream_t streamId)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSetStream((cusparseHandle_t)handle, (cudaStream_t)streamId);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSgebsr2csr(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nb, const UPTKsparseMatDescr_t descrA, const float *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDim, int colBlockDim, const UPTKsparseMatDescr_t descrC, float *csrSortedValC, int *csrSortedRowPtrC, int *csrSortedColIndC)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSgebsr2csr((cusparseHandle_t)handle, cuda_dirA, mb, nb, (const cusparseMatDescr_t)descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDim, colBlockDim, (const cusparseMatDescr_t)descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSgebsr2gebsc(UPTKsparseHandle_t handle, int mb, int nb, int nnzb, const float *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int rowBlockDim, int colBlockDim, float *bscVal, int *bscRowInd, int *bscColPtr, UPTKsparseAction_t copyValues, UPTKsparseIndexBase_t idxBase, void *pBuffer)
{
    cusparseAction_t cuda_copy_values = UPTKsparseActionTocusparseAction(copyValues);
    cusparseIndexBase_t cuda_idx_base = UPTKsparseIndexBaseTocusparseIndexBase(idxBase);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSgebsr2gebsc((cusparseHandle_t)handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim, bscVal, bscRowInd, bscColPtr, cuda_copy_values, cuda_idx_base, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSgebsr2gebsc_bufferSize(UPTKsparseHandle_t handle, int mb, int nb, int nnzb, const float *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int rowBlockDim, int colBlockDim, int *pBufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSgebsr2gebsc_bufferSize((cusparseHandle_t)handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSgebsr2gebsr(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nb, int nnzb, const UPTKsparseMatDescr_t descrA, const float *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, const UPTKsparseMatDescr_t descrC, float *bsrSortedValC, int *bsrSortedRowPtrC, int *bsrSortedColIndC, int rowBlockDimC, int colBlockDimC, void *pBuffer)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSgebsr2gebsr((cusparseHandle_t)handle, cuda_dirA, mb, nb, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA, colBlockDimA, (const cusparseMatDescr_t)descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC, rowBlockDimC, colBlockDimC, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSgebsr2gebsr_bufferSize(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nb, int nnzb, const UPTKsparseMatDescr_t descrA, const float *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, int rowBlockDimC, int colBlockDimC, int *pBufferSizeInBytes)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSgebsr2gebsr_bufferSize((cusparseHandle_t)handle, cuda_dirA, mb, nb, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSgemvi(UPTKsparseHandle_t handle, UPTKsparseOperation_t transA, int m, int n, const float *alpha, const float *A, int lda, int nnz, const float *xVal, const int *xInd, const float *beta, float *y, UPTKsparseIndexBase_t idxBase, void *pBuffer)
{
    cusparseOperation_t cuda_transA = UPTKsparseOperationTocusparseOperation(transA);
    cusparseIndexBase_t cuda_idxBase = UPTKsparseIndexBaseTocusparseIndexBase(idxBase);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSgemvi((cusparseHandle_t)handle, cuda_transA, m, n, alpha, A, lda, nnz, xVal, xInd, beta, y, cuda_idxBase, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSgemvi_bufferSize(UPTKsparseHandle_t handle, UPTKsparseOperation_t transA, int m, int n, int nnz, int *pBufferSize)
{
    cusparseOperation_t cuda_transA = UPTKsparseOperationTocusparseOperation(transA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSgemvi_bufferSize((cusparseHandle_t)handle, cuda_transA, m, n, nnz, pBufferSize);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSgpsvInterleavedBatch(UPTKsparseHandle_t handle, int algo, int m, float *ds, float *dl, float *d, float *du, float *dw, float *x, int batchCount, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSgpsvInterleavedBatch((cusparseHandle_t)handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSgpsvInterleavedBatch_bufferSizeExt(UPTKsparseHandle_t handle, int algo, int m, const float *ds, const float *dl, const float *d, const float *du, const float *dw, const float *x, int batchCount, size_t *pBufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSgpsvInterleavedBatch_bufferSizeExt((cusparseHandle_t)handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSgtsv2(UPTKsparseHandle_t handle, int m, int n, const float *dl, const float *d, const float *du, float *B, int ldb, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSgtsv2((cusparseHandle_t)handle, m, n, dl, d, du, B, ldb, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSgtsv2StridedBatch(UPTKsparseHandle_t handle, int m, const float *dl, const float *d, const float *du, float *x, int batchCount, int batchStride, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSgtsv2StridedBatch((cusparseHandle_t)handle, m, dl, d, du, x, batchCount, batchStride, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSgtsv2StridedBatch_bufferSizeExt(UPTKsparseHandle_t handle, int m, const float *dl, const float *d, const float *du, const float *x, int batchCount, int batchStride, size_t *bufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSgtsv2StridedBatch_bufferSizeExt((cusparseHandle_t)handle, m, dl, d, du, x, batchCount, batchStride, bufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSgtsv2_bufferSizeExt(UPTKsparseHandle_t handle, int m, int n, const float *dl, const float *d, const float *du, const float *B, int ldb, size_t *bufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSgtsv2_bufferSizeExt((cusparseHandle_t)handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSgtsv2_nopivot(UPTKsparseHandle_t handle, int m, int n, const float *dl, const float *d, const float *du, float *B, int ldb, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSgtsv2_nopivot((cusparseHandle_t)handle, m, n, dl, d, du, B, ldb, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSgtsv2_nopivot_bufferSizeExt(UPTKsparseHandle_t handle, int m, int n, const float *dl, const float *d, const float *du, const float *B, int ldb, size_t *bufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSgtsv2_nopivot_bufferSizeExt((cusparseHandle_t)handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSgtsvInterleavedBatch(UPTKsparseHandle_t handle, int algo, int m, float *dl, float *d, float *du, float *x, int batchCount, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSgtsvInterleavedBatch((cusparseHandle_t)handle, algo, m, dl, d, du, x, batchCount, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSgtsvInterleavedBatch_bufferSizeExt(UPTKsparseHandle_t handle, int algo, int m, const float *dl, const float *d, const float *du, const float *x, int batchCount, size_t *pBufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSgtsvInterleavedBatch_bufferSizeExt((cusparseHandle_t)handle, algo, m, dl, d, du, x, batchCount, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSnnz(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int m, int n, const UPTKsparseMatDescr_t descrA, const float *A, int lda, int *nnzPerRowCol, int *nnzTotalDevHostPtr)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSnnz((cusparseHandle_t)handle, cuda_dirA, m, n, (const cusparseMatDescr_t)descrA, A, lda, nnzPerRowCol, nnzTotalDevHostPtr);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSnnz_compress(UPTKsparseHandle_t handle, int m, const UPTKsparseMatDescr_t descr, const float *csrSortedValA, const int *csrSortedRowPtrA, int *nnzPerRow, int *nnzC, float tol)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSnnz_compress((cusparseHandle_t)handle, m, (const cusparseMatDescr_t)descr, csrSortedValA, csrSortedRowPtrA, nnzPerRow, nnzC, tol);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSpMM(UPTKsparseHandle_t handle, UPTKsparseOperation_t opA, UPTKsparseOperation_t opB, const void *alpha, const UPTKsparseSpMatDescr_t matA, const UPTKsparseDnMatDescr_t matB, const void *beta, UPTKsparseDnMatDescr_t matC, UPTKDataType computeType, UPTKsparseSpMMAlg_t alg, void *externalBuffer)
{
    cusparseOperation_t cuda_opA = UPTKsparseOperationTocusparseOperation(opA);
    cusparseOperation_t cuda_opB = UPTKsparseOperationTocusparseOperation(opB);
    cudaDataType cuda_computeType = UPTKDataTypeTocudaDataType(computeType);
    cusparseSpMMAlg_t cuda_alg = UPTKsparseSpMMAlgTocusparseSpMMAlg(alg);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSpMM((cusparseHandle_t)handle, cuda_opA, cuda_opB, alpha, (const cusparseSpMatDescr_t)matA, (const cusparseDnMatDescr_t)matB, beta, (const cusparseDnMatDescr_t)matC, cuda_computeType, cuda_alg, externalBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSpMM_bufferSize(UPTKsparseHandle_t handle, UPTKsparseOperation_t opA, UPTKsparseOperation_t opB, const void *alpha, const UPTKsparseSpMatDescr_t matA, const UPTKsparseDnMatDescr_t matB, const void *beta, UPTKsparseDnMatDescr_t matC, UPTKDataType computeType, UPTKsparseSpMMAlg_t alg, size_t *bufferSize)
{
    cusparseOperation_t cuda_opA = UPTKsparseOperationTocusparseOperation(opA);
    cusparseOperation_t cuda_opB = UPTKsparseOperationTocusparseOperation(opB);
    cudaDataType cuda_computeType = UPTKDataTypeTocudaDataType(computeType);
    cusparseSpMMAlg_t cuda_alg = UPTKsparseSpMMAlgTocusparseSpMMAlg(alg);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSpMM_bufferSize((cusparseHandle_t)handle, cuda_opA, cuda_opB, alpha, (const cusparseSpMatDescr_t)matA, (const cusparseDnMatDescr_t)matB, beta, (const cusparseDnMatDescr_t)matC, cuda_computeType, cuda_alg, bufferSize);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSpMV(UPTKsparseHandle_t handle, UPTKsparseOperation_t opA, const void *alpha, const UPTKsparseSpMatDescr_t matA, const UPTKsparseDnVecDescr_t vecX, const void *beta, const UPTKsparseDnVecDescr_t vecY, UPTKDataType computeType, UPTKsparseSpMVAlg_t alg, void *externalBuffer)
{
    cusparseOperation_t cuda_opA = UPTKsparseOperationTocusparseOperation(opA);
    cudaDataType cuda_computeType = UPTKDataTypeTocudaDataType(computeType);
    cusparseSpMVAlg_t cuda_alg = UPTKsparseSpMVAlgTocusparseSpMVAlg(alg);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSpMV((cusparseHandle_t)handle, cuda_opA, alpha, (const cusparseSpMatDescr_t)matA, (const cusparseDnVecDescr_t)vecX, beta, (const cusparseDnVecDescr_t)vecY, cuda_computeType, cuda_alg, externalBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSpMV_bufferSize(UPTKsparseHandle_t handle, UPTKsparseOperation_t opA, const void *alpha, const UPTKsparseSpMatDescr_t matA, const UPTKsparseDnVecDescr_t vecX, const void *beta, const UPTKsparseDnVecDescr_t vecY, UPTKDataType computeType, UPTKsparseSpMVAlg_t alg, size_t *bufferSize)
{
    cusparseOperation_t cuda_opA = UPTKsparseOperationTocusparseOperation(opA);
    cudaDataType cuda_computeType = UPTKDataTypeTocudaDataType(computeType);
    cusparseSpMVAlg_t cuda_alg = UPTKsparseSpMVAlgTocusparseSpMVAlg(alg);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSpMV_bufferSize((cusparseHandle_t)handle, cuda_opA, alpha, (const cusparseSpMatDescr_t)matA, (const cusparseDnVecDescr_t)vecX, beta, (const cusparseDnVecDescr_t)vecY, cuda_computeType, cuda_alg, bufferSize);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSpMatGetFormat(const UPTKsparseSpMatDescr_t spMatDescr, UPTKsparseFormat_t *format)
{
    cusparseFormat_t cuda_format;
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSpMatGetFormat((const cusparseSpMatDescr_t)spMatDescr, &cuda_format);
    *format = cusparseFormatToUPTKsparseFormat(cuda_format);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSpMatGetIndexBase(const UPTKsparseSpMatDescr_t spMatDescr, UPTKsparseIndexBase_t *idxBase)
{
    cusparseIndexBase_t cuda_idxBase;
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSpMatGetIndexBase((const cusparseSpMatDescr_t)spMatDescr, &cuda_idxBase);
    *idxBase = cusparseIndexBaseToUPTKsparseIndexBase(cuda_idxBase);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

// UNSUPPORTED
UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSpMatGetStridedBatch(const UPTKsparseSpMatDescr_t spMatDescr, int *batchCount)
{
    return cusparseStatusToUPTKsparseStatus(cusparseSpMatGetStridedBatch((cusparseSpMatDescr_t)spMatDescr, batchCount));
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSpMatGetValues(const UPTKsparseSpMatDescr_t spMatDescr, void **values)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSpMatGetValues((cusparseSpMatDescr_t)spMatDescr, values);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSpMatSetValues(UPTKsparseSpMatDescr_t spMatDescr, void *values)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSpMatSetValues((cusparseSpMatDescr_t)spMatDescr, values);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSpVV(UPTKsparseHandle_t handle, UPTKsparseOperation_t opX, const UPTKsparseSpVecDescr_t vecX, const UPTKsparseDnVecDescr_t vecY, void *result, UPTKDataType computeType, void *externalBuffer)
{
    cusparseOperation_t cuda_opX = UPTKsparseOperationTocusparseOperation(opX);
    cudaDataType cuda_computeType = UPTKDataTypeTocudaDataType(computeType);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSpVV((cusparseHandle_t)handle, cuda_opX, (cusparseSpVecDescr_t)vecX, (cusparseDnVecDescr_t)vecY, result, cuda_computeType, externalBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSpVV_bufferSize(UPTKsparseHandle_t handle, UPTKsparseOperation_t opX, const UPTKsparseSpVecDescr_t vecX, const UPTKsparseDnVecDescr_t vecY, const void *result, UPTKDataType computeType, size_t *bufferSize)
{
    cusparseOperation_t cuda_opX = UPTKsparseOperationTocusparseOperation(opX);
    cudaDataType cuda_computeType = UPTKDataTypeTocudaDataType(computeType);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSpVV_bufferSize((cusparseHandle_t)handle, cuda_opX, (cusparseSpVecDescr_t)vecX, (cusparseDnVecDescr_t)vecY, const_cast<void *>(result), cuda_computeType, bufferSize);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSpVecGet(const UPTKsparseSpVecDescr_t spVecDescr, int64_t *size, int64_t *nnz, void **indices, void **values, UPTKsparseIndexType_t *idxType, UPTKsparseIndexBase_t *idxBase, UPTKDataType *valueType)
{
    cusparseIndexType_t cuda_idxType;
    cusparseIndexBase_t cuda_idxBase;
    cudaDataType cuda_valueType;
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSpVecGet((const cusparseSpVecDescr_t)spVecDescr, size, nnz, indices, values, &cuda_idxType, &cuda_idxBase, &cuda_valueType);
    *idxType = cusparseIndexTypeToUPTKsparseIndexType(cuda_idxType);
    *idxBase = cusparseIndexBaseToUPTKsparseIndexBase(cuda_idxBase);
    *valueType = cudaDataTypeToUPTKDataType(cuda_valueType);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSpVecGetIndexBase(const UPTKsparseSpVecDescr_t spVecDescr, UPTKsparseIndexBase_t *idxBase)
{
    cusparseIndexBase_t cuda_idxBase;
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSpVecGetIndexBase((const cusparseSpVecDescr_t)spVecDescr, &cuda_idxBase);
    *idxBase = cusparseIndexBaseToUPTKsparseIndexBase(cuda_idxBase);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSpVecGetValues(const UPTKsparseSpVecDescr_t spVecDescr, void **values)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSpVecGetValues((const cusparseSpVecDescr_t)spVecDescr, values);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSpVecSetValues(UPTKsparseSpVecDescr_t spVecDescr, void *values)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSpVecSetValues((cusparseSpVecDescr_t)spVecDescr, values);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSpruneCsr2csr(UPTKsparseHandle_t handle, int m, int n, int nnzA, const UPTKsparseMatDescr_t descrA, const float *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, const float *threshold, const UPTKsparseMatDescr_t descrC, float *csrSortedValC, const int *csrSortedRowPtrC, int *csrSortedColIndC, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSpruneCsr2csr((cusparseHandle_t)handle, m, n, nnzA, (const cusparseMatDescr_t)descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, threshold, (const cusparseMatDescr_t)descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSpruneCsr2csrByPercentage(UPTKsparseHandle_t handle, int m, int n, int nnzA, const UPTKsparseMatDescr_t descrA, const float *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage, const UPTKsparseMatDescr_t descrC, float *csrSortedValC, const int *csrSortedRowPtrC, int *csrSortedColIndC, pruneInfo_t info, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSpruneCsr2csrByPercentage((cusparseHandle_t)handle, m, n, nnzA, (const cusparseMatDescr_t)descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, percentage, (const cusparseMatDescr_t)descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, (pruneInfo_t)info, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSpruneCsr2csrByPercentage_bufferSizeExt(UPTKsparseHandle_t handle, int m, int n, int nnzA, const UPTKsparseMatDescr_t descrA, const float *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage, const UPTKsparseMatDescr_t descrC, const float *csrSortedValC, const int *csrSortedRowPtrC, const int *csrSortedColIndC, pruneInfo_t info, size_t *pBufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSpruneCsr2csrByPercentage_bufferSizeExt((cusparseHandle_t)handle, m, n, nnzA, (const cusparseMatDescr_t)descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, percentage, (const cusparseMatDescr_t)descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, (pruneInfo_t)info, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSpruneCsr2csrNnz(UPTKsparseHandle_t handle, int m, int n, int nnzA, const UPTKsparseMatDescr_t descrA, const float *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, const float *threshold, const UPTKsparseMatDescr_t descrC, int *csrSortedRowPtrC, int *nnzTotalDevHostPtr, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSpruneCsr2csrNnz((cusparseHandle_t)handle, m, n, nnzA, (const cusparseMatDescr_t)descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, threshold, (const cusparseMatDescr_t)descrC, csrSortedRowPtrC, nnzTotalDevHostPtr, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSpruneCsr2csrNnzByPercentage(UPTKsparseHandle_t handle, int m, int n, int nnzA, const UPTKsparseMatDescr_t descrA, const float *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage, const UPTKsparseMatDescr_t descrC, int *csrSortedRowPtrC, int *nnzTotalDevHostPtr, pruneInfo_t info, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSpruneCsr2csrNnzByPercentage((cusparseHandle_t)handle, m, n, nnzA, (const cusparseMatDescr_t)descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, percentage, (const cusparseMatDescr_t)descrC, csrSortedRowPtrC, nnzTotalDevHostPtr, (pruneInfo_t)info, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSpruneCsr2csr_bufferSizeExt(UPTKsparseHandle_t handle, int m, int n, int nnzA, const UPTKsparseMatDescr_t descrA, const float *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, const float *threshold, const UPTKsparseMatDescr_t descrC, const float *csrSortedValC, const int *csrSortedRowPtrC, const int *csrSortedColIndC, size_t *pBufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSpruneCsr2csr_bufferSizeExt((cusparseHandle_t)handle, m, n, nnzA, (const cusparseMatDescr_t)descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, threshold, (const cusparseMatDescr_t)descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSpruneDense2csr(UPTKsparseHandle_t handle, int m, int n, const float *A, int lda, const float *threshold, const UPTKsparseMatDescr_t descrC, float *csrSortedValC, const int *csrSortedRowPtrC, int *csrSortedColIndC, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSpruneDense2csr((cusparseHandle_t)handle, m, n, A, lda, threshold, (const cusparseMatDescr_t)descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSpruneDense2csrByPercentage(UPTKsparseHandle_t handle, int m, int n, const float *A, int lda, float percentage, const UPTKsparseMatDescr_t descrC, float *csrSortedValC, const int *csrSortedRowPtrC, int *csrSortedColIndC, pruneInfo_t info, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSpruneDense2csrByPercentage((cusparseHandle_t)handle, m, n, A, lda, percentage, (const cusparseMatDescr_t)descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, (pruneInfo_t)info, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSpruneDense2csrByPercentage_bufferSizeExt(UPTKsparseHandle_t handle, int m, int n, const float *A, int lda, float percentage, const UPTKsparseMatDescr_t descrC, const float *csrSortedValC, const int *csrSortedRowPtrC, const int *csrSortedColIndC, pruneInfo_t info, size_t *pBufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSpruneDense2csrByPercentage_bufferSizeExt((cusparseHandle_t)handle, m, n, A, lda, percentage, (const cusparseMatDescr_t)descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, (pruneInfo_t)info, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSpruneDense2csrNnz(UPTKsparseHandle_t handle, int m, int n, const float *A, int lda, const float *threshold, const UPTKsparseMatDescr_t descrC, int *csrRowPtrC, int *nnzTotalDevHostPtr, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSpruneDense2csrNnz((cusparseHandle_t)handle, m, n, A, lda, threshold, (const cusparseMatDescr_t)descrC, csrRowPtrC, nnzTotalDevHostPtr, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSpruneDense2csrNnzByPercentage(UPTKsparseHandle_t handle, int m, int n, const float *A, int lda, float percentage, const UPTKsparseMatDescr_t descrC, int *csrRowPtrC, int *nnzTotalDevHostPtr, pruneInfo_t info, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSpruneDense2csrNnzByPercentage((cusparseHandle_t)handle, m, n, A, lda, percentage, (const cusparseMatDescr_t)descrC, csrRowPtrC, nnzTotalDevHostPtr, (pruneInfo_t)info, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSpruneDense2csr_bufferSizeExt(UPTKsparseHandle_t handle, int m, int n, const float *A, int lda, const float *threshold, const UPTKsparseMatDescr_t descrC, const float *csrSortedValC, const int *csrSortedRowPtrC, const int *csrSortedColIndC, size_t *pBufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSpruneDense2csr_bufferSizeExt((cusparseHandle_t)handle, m, n, A, lda, threshold, (const cusparseMatDescr_t)descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseXbsric02_zeroPivot(UPTKsparseHandle_t handle, bsric02Info_t info, int *position)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseXbsric02_zeroPivot((cusparseHandle_t)handle, (bsric02Info_t)info, position);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseXbsrilu02_zeroPivot(UPTKsparseHandle_t handle, bsrilu02Info_t info, int *position)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseXbsrilu02_zeroPivot((cusparseHandle_t)handle, (bsrilu02Info_t)info, position);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseXbsrsm2_zeroPivot(UPTKsparseHandle_t handle, bsrsm2Info_t info, int *position)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseXbsrsm2_zeroPivot((cusparseHandle_t)handle, (bsrsm2Info_t)info, position);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseXbsrsv2_zeroPivot(UPTKsparseHandle_t handle, bsrsv2Info_t info, int *position)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseXbsrsv2_zeroPivot((cusparseHandle_t)handle, (bsrsv2Info_t)info, position);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseXcoo2csr(UPTKsparseHandle_t handle, const int *cooRowInd, int nnz, int m, int *csrSortedRowPtr, UPTKsparseIndexBase_t idxBase)
{
    cusparseIndexBase_t cuda_idxBase = UPTKsparseIndexBaseTocusparseIndexBase(idxBase);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseXcoo2csr((cusparseHandle_t)handle, cooRowInd, nnz, m, csrSortedRowPtr, cuda_idxBase);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseXcoosortByColumn(UPTKsparseHandle_t handle, int m, int n, int nnz, int *cooRowsA, int *cooColsA, int *P, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseXcoosortByColumn((cusparseHandle_t)handle, m, n, nnz, cooRowsA, cooColsA, P, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseXcoosortByRow(UPTKsparseHandle_t handle, int m, int n, int nnz, int *cooRowsA, int *cooColsA, int *P, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseXcoosortByRow((cusparseHandle_t)handle, m, n, nnz, cooRowsA, cooColsA, P, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseXcoosort_bufferSizeExt(UPTKsparseHandle_t handle, int m, int n, int nnz, const int *cooRowsA, const int *cooColsA, size_t *pBufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseXcoosort_bufferSizeExt((cusparseHandle_t)handle, m, n, nnz, cooRowsA, cooColsA, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseXcscsort(UPTKsparseHandle_t handle, int m, int n, int nnz, const UPTKsparseMatDescr_t descrA, const int *cscColPtrA, int *cscRowIndA, int *P, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseXcscsort((cusparseHandle_t)handle, m, n, nnz, (const cusparseMatDescr_t)descrA, cscColPtrA, cscRowIndA, P, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseXcscsort_bufferSizeExt(UPTKsparseHandle_t handle, int m, int n, int nnz, const int *cscColPtrA, const int *cscRowIndA, size_t *pBufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseXcscsort_bufferSizeExt((cusparseHandle_t)handle, m, n, nnz, cscColPtrA, cscRowIndA, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseXcsr2bsrNnz(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int m, int n, const UPTKsparseMatDescr_t descrA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, int blockDim, const UPTKsparseMatDescr_t descrC, int *bsrSortedRowPtrC, int *nnzTotalDevHostPtr)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseXcsr2bsrNnz((cusparseHandle_t)handle, cuda_dirA, m, n, (const cusparseMatDescr_t)descrA, csrSortedRowPtrA, csrSortedColIndA, blockDim, (const cusparseMatDescr_t)descrC, bsrSortedRowPtrC, nnzTotalDevHostPtr);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseXcsr2coo(UPTKsparseHandle_t handle, const int *csrSortedRowPtr, int nnz, int m, int *cooRowInd, UPTKsparseIndexBase_t idxBase)
{
    cusparseIndexBase_t cuda_idxBase = UPTKsparseIndexBaseTocusparseIndexBase(idxBase);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseXcsr2coo((cusparseHandle_t)handle, csrSortedRowPtr, nnz, m, cooRowInd, cuda_idxBase);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseXcsr2gebsrNnz(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int m, int n, const UPTKsparseMatDescr_t descrA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, const UPTKsparseMatDescr_t descrC, int *bsrSortedRowPtrC, int rowBlockDim, int colBlockDim, int *nnzTotalDevHostPtr, void *pBuffer)
{
    cusparseDirection_t cuda_dir = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseXcsr2gebsrNnz((cusparseHandle_t)handle, cuda_dir, m, n, (const cusparseMatDescr_t)descrA, csrSortedRowPtrA, csrSortedColIndA, (const cusparseMatDescr_t)descrC, bsrSortedRowPtrC, rowBlockDim, colBlockDim, nnzTotalDevHostPtr, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseXcsrgeam2Nnz(UPTKsparseHandle_t handle, int m, int n, const UPTKsparseMatDescr_t descrA, int nnzA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, const UPTKsparseMatDescr_t descrB, int nnzB, const int *csrSortedRowPtrB, const int *csrSortedColIndB, const UPTKsparseMatDescr_t descrC, int *csrSortedRowPtrC, int *nnzTotalDevHostPtr, void *workspace)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseXcsrgeam2Nnz((cusparseHandle_t)handle, m, n, (const cusparseMatDescr_t)descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, (const cusparseMatDescr_t)descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB, (const cusparseMatDescr_t)descrC, csrSortedRowPtrC, nnzTotalDevHostPtr, workspace);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseXcsric02_zeroPivot(UPTKsparseHandle_t handle, csric02Info_t info, int *position)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseXcsric02_zeroPivot((cusparseHandle_t)handle, (csric02Info_t)info, position);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseXcsrilu02_zeroPivot(UPTKsparseHandle_t handle, csrilu02Info_t info, int *position)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseXcsrilu02_zeroPivot((cusparseHandle_t)handle, (csrilu02Info_t)info, position);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseXcsrsort(UPTKsparseHandle_t handle, int m, int n, int nnz, const UPTKsparseMatDescr_t descrA, const int *csrRowPtrA, int *csrColIndA, int *P, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseXcsrsort((cusparseHandle_t)handle, m, n, nnz, (const cusparseMatDescr_t)descrA, csrRowPtrA, csrColIndA, P, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseXcsrsort_bufferSizeExt(UPTKsparseHandle_t handle, int m, int n, int nnz, const int *csrRowPtrA, const int *csrColIndA, size_t *pBufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseXcsrsort_bufferSizeExt((cusparseHandle_t)handle, m, n, nnz, csrRowPtrA, csrColIndA, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseXgebsr2gebsrNnz(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nb, int nnzb, const UPTKsparseMatDescr_t descrA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, const UPTKsparseMatDescr_t descrC, int *bsrSortedRowPtrC, int rowBlockDimC, int colBlockDimC, int *nnzTotalDevHostPtr, void *pBuffer)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseXgebsr2gebsrNnz((cusparseHandle_t)handle, cuda_dirA, mb, nb, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA, colBlockDimA, (const cusparseMatDescr_t)descrC, bsrSortedRowPtrC, rowBlockDimC, colBlockDimC, nnzTotalDevHostPtr, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZbsr2csr(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nb, const UPTKsparseMatDescr_t descrA, const cuDoubleComplex *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim, const UPTKsparseMatDescr_t descrC, cuDoubleComplex *csrSortedValC, int *csrSortedRowPtrC, int *csrSortedColIndC)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZbsr2csr((cusparseHandle_t)handle, cuda_dirA, mb, nb, (const cusparseMatDescr_t)descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, (const cusparseMatDescr_t)descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZbsric02(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nnzb, const UPTKsparseMatDescr_t descrA, cuDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim, bsric02Info_t info, UPTKsparseSolvePolicy_t policy, void *pBuffer)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseSolvePolicy_t cuda_policy = UPTKsparseSolvePolicyTocusparseSolvePolicy(policy);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZbsric02((cusparseHandle_t)handle, cuda_dirA, mb, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, (bsric02Info_t)info, cuda_policy, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZbsric02_analysis(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nnzb, const UPTKsparseMatDescr_t descrA, const cuDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim, bsric02Info_t info, UPTKsparseSolvePolicy_t policy, void *pInputBuffer)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseSolvePolicy_t cuda_policy = UPTKsparseSolvePolicyTocusparseSolvePolicy(policy);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZbsric02_analysis((cusparseHandle_t)handle, cuda_dirA, mb, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, (bsric02Info_t)info, cuda_policy, pInputBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZbsric02_bufferSize(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nnzb, const UPTKsparseMatDescr_t descrA, cuDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim, bsric02Info_t info, int *pBufferSizeInBytes)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZbsric02_bufferSize((cusparseHandle_t)handle, cuda_dirA, mb, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, (bsric02Info_t)info, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZbsrilu02(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nnzb, const UPTKsparseMatDescr_t descrA, cuDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim, bsrilu02Info_t info, UPTKsparseSolvePolicy_t policy, void *pBuffer)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseSolvePolicy_t cuda_policy = UPTKsparseSolvePolicyTocusparseSolvePolicy(policy);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZbsrilu02((cusparseHandle_t)handle, cuda_dirA, mb, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, (bsrilu02Info_t)info, cuda_policy, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZbsrilu02_analysis(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nnzb, const UPTKsparseMatDescr_t descrA, cuDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim, bsrilu02Info_t info, UPTKsparseSolvePolicy_t policy, void *pBuffer)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseSolvePolicy_t cuda_policy = UPTKsparseSolvePolicyTocusparseSolvePolicy(policy);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZbsrilu02_analysis((cusparseHandle_t)handle, cuda_dirA, mb, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, (bsrilu02Info_t)info, cuda_policy, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZbsrilu02_bufferSize(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nnzb, const UPTKsparseMatDescr_t descrA, cuDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockDim, bsrilu02Info_t info, int *pBufferSizeInBytes)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZbsrilu02_bufferSize((cusparseHandle_t)handle, cuda_dirA, mb, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockDim, (bsrilu02Info_t)info, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZbsrilu02_numericBoost(UPTKsparseHandle_t handle, bsrilu02Info_t info, int enable_boost, double *tol, cuDoubleComplex *boost_val)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZbsrilu02_numericBoost((cusparseHandle_t)handle, (bsrilu02Info_t)info, enable_boost, tol, boost_val);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZbsrmm(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, UPTKsparseOperation_t transA, UPTKsparseOperation_t transB, int mb, int n, int kb, int nnzb, const cuDoubleComplex *alpha, const UPTKsparseMatDescr_t descrA, const cuDoubleComplex *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, const int blockSize, const cuDoubleComplex *B, const int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseOperation_t cuda_transA = UPTKsparseOperationTocusparseOperation(transA);
    cusparseOperation_t cuda_transB = UPTKsparseOperationTocusparseOperation(transB);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZbsrmm((cusparseHandle_t)handle, cuda_dirA, cuda_transA, cuda_transB, mb, n, kb, nnzb, alpha, (const cusparseMatDescr_t)descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockSize, B, ldb, beta, C, ldc);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZbsrmv(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, UPTKsparseOperation_t transA, int mb, int nb, int nnzb, const cuDoubleComplex *alpha, const UPTKsparseMatDescr_t descrA, const cuDoubleComplex *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim, const cuDoubleComplex *x, const cuDoubleComplex *beta, cuDoubleComplex *y)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseOperation_t cuda_transA = UPTKsparseOperationTocusparseOperation(transA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZbsrmv((cusparseHandle_t)handle, cuda_dirA, cuda_transA, mb, nb, nnzb, alpha, (const cusparseMatDescr_t)descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, x, beta, y);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZbsrsm2_analysis(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, UPTKsparseOperation_t transA, UPTKsparseOperation_t transXY, int mb, int n, int nnzb, const UPTKsparseMatDescr_t descrA, const cuDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info, UPTKsparseSolvePolicy_t policy, void *pBuffer)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseOperation_t cuda_transA = UPTKsparseOperationTocusparseOperation(transA);
    cusparseOperation_t cuda_transX = UPTKsparseOperationTocusparseOperation(transXY);
    cusparseSolvePolicy_t cuda_policy = UPTKsparseSolvePolicyTocusparseSolvePolicy(policy);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZbsrsm2_analysis((cusparseHandle_t)handle, cuda_dirA, cuda_transA, cuda_transX, mb, n, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, (bsrsm2Info_t)info, cuda_policy, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZbsrsm2_bufferSize(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, UPTKsparseOperation_t transA, UPTKsparseOperation_t transXY, int mb, int n, int nnzb, const UPTKsparseMatDescr_t descrA, cuDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info, int *pBufferSizeInBytes)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseOperation_t cuda_transA = UPTKsparseOperationTocusparseOperation(transA);
    cusparseOperation_t cuda_transX = UPTKsparseOperationTocusparseOperation(transXY);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZbsrsm2_bufferSize((cusparseHandle_t)handle, cuda_dirA, cuda_transA, cuda_transX, mb, n, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, (bsrsm2Info_t)info, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZbsrsm2_solve(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, UPTKsparseOperation_t transA, UPTKsparseOperation_t transXY, int mb, int n, int nnzb, const cuDoubleComplex *alpha, const UPTKsparseMatDescr_t descrA, const cuDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info, const cuDoubleComplex *B, int ldb, cuDoubleComplex *X, int ldx, UPTKsparseSolvePolicy_t policy, void *pBuffer)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseOperation_t cuda_transA = UPTKsparseOperationTocusparseOperation(transA);
    cusparseOperation_t cuda_transX = UPTKsparseOperationTocusparseOperation(transXY);
    cusparseSolvePolicy_t cuda_policy = UPTKsparseSolvePolicyTocusparseSolvePolicy(policy);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZbsrsm2_solve((cusparseHandle_t)handle, cuda_dirA, cuda_transA, cuda_transX, mb, n, nnzb, alpha, (const cusparseMatDescr_t)descrA, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize, (bsrsm2Info_t)info, B, ldb, X, ldx, cuda_policy, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZbsrsv2_analysis(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, UPTKsparseOperation_t transA, int mb, int nnzb, const UPTKsparseMatDescr_t descrA, const cuDoubleComplex *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim, bsrsv2Info_t info, UPTKsparseSolvePolicy_t policy, void *pBuffer)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseOperation_t cuda_transA = UPTKsparseOperationTocusparseOperation(transA);
    cusparseSolvePolicy_t cuda_policy = UPTKsparseSolvePolicyTocusparseSolvePolicy(policy);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZbsrsv2_analysis((cusparseHandle_t)handle, cuda_dirA, cuda_transA, mb, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, (bsrsv2Info_t)info, cuda_policy, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZbsrsv2_bufferSize(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, UPTKsparseOperation_t transA, int mb, int nnzb, const UPTKsparseMatDescr_t descrA, cuDoubleComplex *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim, bsrsv2Info_t info, int *pBufferSizeInBytes)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseOperation_t cuda_transA = UPTKsparseOperationTocusparseOperation(transA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZbsrsv2_bufferSize((cusparseHandle_t)handle, cuda_dirA, cuda_transA, mb, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, (bsrsv2Info_t)info, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZbsrsv2_bufferSizeExt(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, UPTKsparseOperation_t transA, int mb, int nnzb, const UPTKsparseMatDescr_t descrA, cuDoubleComplex *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockSize, bsrsv2Info_t info, size_t *pBufferSize)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseOperation_t cuda_transA = UPTKsparseOperationTocusparseOperation(transA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZbsrsv2_bufferSizeExt((cusparseHandle_t)handle, cuda_dirA, cuda_transA, mb, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockSize, (bsrsv2Info_t)info, pBufferSize);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZbsrsv2_solve(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, UPTKsparseOperation_t transA, int mb, int nnzb, const cuDoubleComplex *alpha, const UPTKsparseMatDescr_t descrA, const cuDoubleComplex *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int blockDim, bsrsv2Info_t info, const cuDoubleComplex *f, cuDoubleComplex *x, UPTKsparseSolvePolicy_t policy, void *pBuffer)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseOperation_t cuda_transA = UPTKsparseOperationTocusparseOperation(transA);
    cusparseSolvePolicy_t cuda_policy = UPTKsparseSolvePolicyTocusparseSolvePolicy(policy);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZbsrsv2_solve((cusparseHandle_t)handle, cuda_dirA, cuda_transA, mb, nnzb, alpha, (const cusparseMatDescr_t)descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim, (bsrsv2Info_t)info, f, x, cuda_policy, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZbsrxmv(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, UPTKsparseOperation_t transA, int sizeOfMask, int mb, int nb, int nnzb, const cuDoubleComplex *alpha, const UPTKsparseMatDescr_t descrA, const cuDoubleComplex *bsrSortedValA, const int *bsrSortedMaskPtrA, const int *bsrSortedRowPtrA, const int *bsrSortedEndPtrA, const int *bsrSortedColIndA, int blockDim, const cuDoubleComplex *x, const cuDoubleComplex *beta, cuDoubleComplex *y)
{
    cusparseDirection_t cuda_dir = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseOperation_t cuda_trans = UPTKsparseOperationTocusparseOperation(transA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZbsrxmv((cusparseHandle_t)handle, cuda_dir, cuda_trans, sizeOfMask, mb, nb, nnzb, alpha, (const cusparseMatDescr_t)descrA, bsrSortedValA, bsrSortedMaskPtrA, bsrSortedRowPtrA, bsrSortedEndPtrA, bsrSortedColIndA, blockDim, x, beta, y);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZcsr2bsr(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int m, int n, const UPTKsparseMatDescr_t descrA, const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, int blockDim, const UPTKsparseMatDescr_t descrC, cuDoubleComplex *bsrSortedValC, int *bsrSortedRowPtrC, int *bsrSortedColIndC)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZcsr2bsr((cusparseHandle_t)handle, cuda_dirA, m, n, (const cusparseMatDescr_t)descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, blockDim, (const cusparseMatDescr_t)descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZcsr2csr_compress(UPTKsparseHandle_t handle, int m, int n, const UPTKsparseMatDescr_t descrA, const cuDoubleComplex *csrSortedValA, const int *csrSortedColIndA, const int *csrSortedRowPtrA, int nnzA, const int *nnzPerRow, cuDoubleComplex *csrSortedValC, int *csrSortedColIndC, int *csrSortedRowPtrC, cuDoubleComplex tol)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZcsr2csr_compress((cusparseHandle_t)handle, m, n, (const cusparseMatDescr_t)descrA, csrSortedValA, csrSortedColIndA, csrSortedRowPtrA, nnzA, nnzPerRow, csrSortedValC, csrSortedColIndC, csrSortedRowPtrC, tol);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZcsr2csru(UPTKsparseHandle_t handle, int m, int n, int nnz, const UPTKsparseMatDescr_t descrA, cuDoubleComplex *csrVal, const int *csrRowPtr, int *csrColInd, csru2csrInfo_t info, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZcsr2csru((cusparseHandle_t)handle, m, n, nnz, (const cusparseMatDescr_t)descrA, csrVal, csrRowPtr, csrColInd, (csru2csrInfo_t)info, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZcsr2gebsr(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int m, int n, const UPTKsparseMatDescr_t descrA, const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, const UPTKsparseMatDescr_t descrC, cuDoubleComplex *bsrSortedValC, int *bsrSortedRowPtrC, int *bsrSortedColIndC, int rowBlockDim, int colBlockDim, void *pBuffer)
{
    cusparseDirection_t cuda_dir = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZcsr2gebsr((cusparseHandle_t)handle, cuda_dir, m, n, (const cusparseMatDescr_t)descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, (const cusparseMatDescr_t)descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC, rowBlockDim, colBlockDim, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZcsr2gebsr_bufferSize(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int m, int n, const UPTKsparseMatDescr_t descrA, const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, int rowBlockDim, int colBlockDim, int *pBufferSizeInBytes)
{
    cusparseDirection_t cuda_dir = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZcsr2gebsr_bufferSize((cusparseHandle_t)handle, cuda_dir, m, n, (const cusparseMatDescr_t)descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, rowBlockDim, colBlockDim, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZcsrcolor(UPTKsparseHandle_t handle, int m, int nnz, const UPTKsparseMatDescr_t descrA, const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, const double *fractionToColor, int *ncolors, int *coloring, int *reordering, const UPTKsparseColorInfo_t info)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZcsrcolor((cusparseHandle_t)handle, m, nnz, (const cusparseMatDescr_t)descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, fractionToColor, ncolors, coloring, reordering, (cusparseColorInfo_t)info);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZcsrgeam2(UPTKsparseHandle_t handle, int m, int n, const cuDoubleComplex *alpha, const UPTKsparseMatDescr_t descrA, int nnzA, const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, const cuDoubleComplex *beta, const UPTKsparseMatDescr_t descrB, int nnzB, const cuDoubleComplex *csrSortedValB, const int *csrSortedRowPtrB, const int *csrSortedColIndB, const UPTKsparseMatDescr_t descrC, cuDoubleComplex *csrSortedValC, int *csrSortedRowPtrC, int *csrSortedColIndC, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZcsrgeam2((cusparseHandle_t)handle, m, n, alpha, (const cusparseMatDescr_t)descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, (const cusparseMatDescr_t)descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, (const cusparseMatDescr_t)descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZcsrgeam2_bufferSizeExt(UPTKsparseHandle_t handle, int m, int n, const cuDoubleComplex *alpha, const UPTKsparseMatDescr_t descrA, int nnzA, const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, const cuDoubleComplex *beta, const UPTKsparseMatDescr_t descrB, int nnzB, const cuDoubleComplex *csrSortedValB, const int *csrSortedRowPtrB, const int *csrSortedColIndB, const UPTKsparseMatDescr_t descrC, const cuDoubleComplex *csrSortedValC, const int *csrSortedRowPtrC, const int *csrSortedColIndC, size_t *pBufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZcsrgeam2_bufferSizeExt((cusparseHandle_t)handle, m, n, alpha, (const cusparseMatDescr_t)descrA, nnzA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, beta, (const cusparseMatDescr_t)descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, (const cusparseMatDescr_t)descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZcsric02(UPTKsparseHandle_t handle, int m, int nnz, const UPTKsparseMatDescr_t descrA, cuDoubleComplex *csrSortedValA_valM, const int *csrSortedRowPtrA, const int *csrSortedColIndA, csric02Info_t info, UPTKsparseSolvePolicy_t policy, void *pBuffer)
{
    cusparseSolvePolicy_t cuda_policy = UPTKsparseSolvePolicyTocusparseSolvePolicy(policy);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZcsric02((cusparseHandle_t)handle, m, nnz, (const cusparseMatDescr_t)descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, (csric02Info_t)info, cuda_policy, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZcsric02_analysis(UPTKsparseHandle_t handle, int m, int nnz, const UPTKsparseMatDescr_t descrA, const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, csric02Info_t info, UPTKsparseSolvePolicy_t policy, void *pBuffer)
{
    cusparseSolvePolicy_t cuda_policy = UPTKsparseSolvePolicyTocusparseSolvePolicy(policy);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZcsric02_analysis((cusparseHandle_t)handle, m, nnz, (const cusparseMatDescr_t)descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, (csric02Info_t)info, cuda_policy, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZcsric02_bufferSize(UPTKsparseHandle_t handle, int m, int nnz, const UPTKsparseMatDescr_t descrA, cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, csric02Info_t info, int *pBufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZcsric02_bufferSize((cusparseHandle_t)handle, m, nnz, (const cusparseMatDescr_t)descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, (csric02Info_t)info, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZcsric02_bufferSizeExt(UPTKsparseHandle_t handle, int m, int nnz, const UPTKsparseMatDescr_t descrA, cuDoubleComplex *csrSortedVal, const int *csrSortedRowPtr, const int *csrSortedColInd, csric02Info_t info, size_t *pBufferSize)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZcsric02_bufferSizeExt((cusparseHandle_t)handle, m, nnz, (const cusparseMatDescr_t)descrA, csrSortedVal, csrSortedRowPtr, csrSortedColInd, (csric02Info_t)info, pBufferSize);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZcsrilu02(UPTKsparseHandle_t handle, int m, int nnz, const UPTKsparseMatDescr_t descrA, cuDoubleComplex *csrSortedValA_valM, const int *csrSortedRowPtrA, const int *csrSortedColIndA, csrilu02Info_t info, UPTKsparseSolvePolicy_t policy, void *pBuffer)
{
    cusparseSolvePolicy_t cuda_policy = UPTKsparseSolvePolicyTocusparseSolvePolicy(policy);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZcsrilu02((cusparseHandle_t)handle, m, nnz, (const cusparseMatDescr_t)descrA, csrSortedValA_valM, csrSortedRowPtrA, csrSortedColIndA, (csrilu02Info_t)info, cuda_policy, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZcsrilu02_analysis(UPTKsparseHandle_t handle, int m, int nnz, const UPTKsparseMatDescr_t descrA, const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, csrilu02Info_t info, UPTKsparseSolvePolicy_t policy, void *pBuffer)
{
    cusparseSolvePolicy_t cuda_policy = UPTKsparseSolvePolicyTocusparseSolvePolicy(policy);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZcsrilu02_analysis((cusparseHandle_t)handle, m, nnz, (const cusparseMatDescr_t)descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, (csrilu02Info_t)info, cuda_policy, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZcsrilu02_bufferSize(UPTKsparseHandle_t handle, int m, int nnz, const UPTKsparseMatDescr_t descrA, cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, csrilu02Info_t info, int *pBufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZcsrilu02_bufferSize((cusparseHandle_t)handle, m, nnz, (const cusparseMatDescr_t)descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, (csrilu02Info_t)info, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZcsrilu02_bufferSizeExt(UPTKsparseHandle_t handle, int m, int nnz, const UPTKsparseMatDescr_t descrA, cuDoubleComplex *csrSortedVal, const int *csrSortedRowPtr, const int *csrSortedColInd, csrilu02Info_t info, size_t *pBufferSize)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZcsrilu02_bufferSizeExt((cusparseHandle_t)handle, m, nnz, (const cusparseMatDescr_t)descrA, csrSortedVal, csrSortedRowPtr, csrSortedColInd, (csrilu02Info_t)info, pBufferSize);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZcsrilu02_numericBoost(UPTKsparseHandle_t handle, csrilu02Info_t info, int enable_boost, double *tol, cuDoubleComplex *boost_val)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZcsrilu02_numericBoost((cusparseHandle_t)handle, (csrilu02Info_t)info, enable_boost, tol, boost_val);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZcsru2csr(UPTKsparseHandle_t handle, int m, int n, int nnz, const UPTKsparseMatDescr_t descrA, cuDoubleComplex *csrVal, const int *csrRowPtr, int *csrColInd, csru2csrInfo_t info, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZcsru2csr((cusparseHandle_t)handle, m, n, nnz, (const cusparseMatDescr_t)descrA, csrVal, csrRowPtr, csrColInd, (csru2csrInfo_t)info, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZcsru2csr_bufferSizeExt(UPTKsparseHandle_t handle, int m, int n, int nnz, cuDoubleComplex *csrVal, const int *csrRowPtr, int *csrColInd, csru2csrInfo_t info, size_t *pBufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZcsru2csr_bufferSizeExt((cusparseHandle_t)handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, (csru2csrInfo_t)info, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZgebsr2csr(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nb, const UPTKsparseMatDescr_t descrA, const cuDoubleComplex *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDim, int colBlockDim, const UPTKsparseMatDescr_t descrC, cuDoubleComplex *csrSortedValC, int *csrSortedRowPtrC, int *csrSortedColIndC)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZgebsr2csr((cusparseHandle_t)handle, cuda_dirA, mb, nb, (const cusparseMatDescr_t)descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDim, colBlockDim, (const cusparseMatDescr_t)descrC, csrSortedValC, csrSortedRowPtrC, csrSortedColIndC);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZgebsr2gebsc(UPTKsparseHandle_t handle, int mb, int nb, int nnzb, const cuDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int rowBlockDim, int colBlockDim, cuDoubleComplex *bscVal, int *bscRowInd, int *bscColPtr, UPTKsparseAction_t copyValues, UPTKsparseIndexBase_t idxBase, void *pBuffer)
{
    cusparseAction_t cuda_copy_values = UPTKsparseActionTocusparseAction(copyValues);
    cusparseIndexBase_t cuda_idx_base = UPTKsparseIndexBaseTocusparseIndexBase(idxBase);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZgebsr2gebsc((cusparseHandle_t)handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim, bscVal, bscRowInd, bscColPtr, cuda_copy_values, cuda_idx_base, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZgebsr2gebsc_bufferSize(UPTKsparseHandle_t handle, int mb, int nb, int nnzb, const cuDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int rowBlockDim, int colBlockDim, int *pBufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZgebsr2gebsc_bufferSize((cusparseHandle_t)handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, rowBlockDim, colBlockDim, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZgebsr2gebsr(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nb, int nnzb, const UPTKsparseMatDescr_t descrA, const cuDoubleComplex *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, const UPTKsparseMatDescr_t descrC, cuDoubleComplex *bsrSortedValC, int *bsrSortedRowPtrC, int *bsrSortedColIndC, int rowBlockDimC, int colBlockDimC, void *pBuffer)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZgebsr2gebsr((cusparseHandle_t)handle, cuda_dirA, mb, nb, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA, colBlockDimA, (const cusparseMatDescr_t)descrC, bsrSortedValC, bsrSortedRowPtrC, bsrSortedColIndC, rowBlockDimC, colBlockDimC, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZgebsr2gebsr_bufferSize(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nb, int nnzb, const UPTKsparseMatDescr_t descrA, const cuDoubleComplex *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, int rowBlockDimC, int colBlockDimC, int *pBufferSizeInBytes)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZgebsr2gebsr_bufferSize((cusparseHandle_t)handle, cuda_dirA, mb, nb, nnzb, (const cusparseMatDescr_t)descrA, bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZgemvi(UPTKsparseHandle_t handle, UPTKsparseOperation_t transA, int m, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda, int nnz, const cuDoubleComplex *xVal, const int *xInd, const cuDoubleComplex *beta, cuDoubleComplex *y, UPTKsparseIndexBase_t idxBase, void *pBuffer)
{
    cusparseOperation_t cuda_transA = UPTKsparseOperationTocusparseOperation(transA);
    cusparseIndexBase_t cuda_idxBase = UPTKsparseIndexBaseTocusparseIndexBase(idxBase);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZgemvi((cusparseHandle_t)handle, cuda_transA, m, n, alpha, A, lda, nnz, xVal, xInd, beta, y, cuda_idxBase, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZgemvi_bufferSize(UPTKsparseHandle_t handle, UPTKsparseOperation_t transA, int m, int n, int nnz, int *pBufferSize)
{
    cusparseOperation_t cuda_transA = UPTKsparseOperationTocusparseOperation(transA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZgemvi_bufferSize((cusparseHandle_t)handle, cuda_transA, m, n, nnz, pBufferSize);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZgpsvInterleavedBatch(UPTKsparseHandle_t handle, int algo, int m, cuDoubleComplex *ds, cuDoubleComplex *dl, cuDoubleComplex *d, cuDoubleComplex *du, cuDoubleComplex *dw, cuDoubleComplex *x, int batchCount, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZgpsvInterleavedBatch((cusparseHandle_t)handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZgpsvInterleavedBatch_bufferSizeExt(UPTKsparseHandle_t handle, int algo, int m, const cuDoubleComplex *ds, const cuDoubleComplex *dl, const cuDoubleComplex *d, const cuDoubleComplex *du, const cuDoubleComplex *dw, const cuDoubleComplex *x, int batchCount, size_t *pBufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZgpsvInterleavedBatch_bufferSizeExt((cusparseHandle_t)handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZgtsv2(UPTKsparseHandle_t handle, int m, int n, const cuDoubleComplex *dl, const cuDoubleComplex *d, const cuDoubleComplex *du, cuDoubleComplex *B, int ldb, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZgtsv2((cusparseHandle_t)handle, m, n, dl, d, du, B, ldb, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZgtsv2StridedBatch(UPTKsparseHandle_t handle, int m, const cuDoubleComplex *dl, const cuDoubleComplex *d, const cuDoubleComplex *du, cuDoubleComplex *x, int batchCount, int batchStride, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZgtsv2StridedBatch((cusparseHandle_t)handle, m, dl, d, du, x, batchCount, batchStride, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZgtsv2StridedBatch_bufferSizeExt(UPTKsparseHandle_t handle, int m, const cuDoubleComplex *dl, const cuDoubleComplex *d, const cuDoubleComplex *du, const cuDoubleComplex *x, int batchCount, int batchStride, size_t *bufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZgtsv2StridedBatch_bufferSizeExt((cusparseHandle_t)handle, m, dl, d, du, x, batchCount, batchStride, bufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZgtsv2_bufferSizeExt(UPTKsparseHandle_t handle, int m, int n, const cuDoubleComplex *dl, const cuDoubleComplex *d, const cuDoubleComplex *du, const cuDoubleComplex *B, int ldb, size_t *bufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZgtsv2_bufferSizeExt((cusparseHandle_t)handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZgtsv2_nopivot(UPTKsparseHandle_t handle, int m, int n, const cuDoubleComplex *dl, const cuDoubleComplex *d, const cuDoubleComplex *du, cuDoubleComplex *B, int ldb, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZgtsv2_nopivot((cusparseHandle_t)handle, m, n, dl, d, du, B, ldb, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZgtsv2_nopivot_bufferSizeExt(UPTKsparseHandle_t handle, int m, int n, const cuDoubleComplex *dl, const cuDoubleComplex *d, const cuDoubleComplex *du, const cuDoubleComplex *B, int ldb, size_t *bufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZgtsv2_nopivot_bufferSizeExt((cusparseHandle_t)handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZgtsvInterleavedBatch(UPTKsparseHandle_t handle, int algo, int m, cuDoubleComplex *dl, cuDoubleComplex *d, cuDoubleComplex *du, cuDoubleComplex *x, int batchCount, void *pBuffer)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZgtsvInterleavedBatch((cusparseHandle_t)handle, algo, m, dl, d, du, x, batchCount, pBuffer);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZgtsvInterleavedBatch_bufferSizeExt(UPTKsparseHandle_t handle, int algo, int m, const cuDoubleComplex *dl, const cuDoubleComplex *d, const cuDoubleComplex *du, const cuDoubleComplex *x, int batchCount, size_t *pBufferSizeInBytes)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZgtsvInterleavedBatch_bufferSizeExt((cusparseHandle_t)handle, algo, m, dl, d, du, x, batchCount, pBufferSizeInBytes);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZnnz(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int m, int n, const UPTKsparseMatDescr_t descrA, const cuDoubleComplex *A, int lda, int *nnzPerRowCol, int *nnzTotalDevHostPtr)
{
    cusparseDirection_t cuda_dirA = UPTKsparseDirectionTocusparseDirection(dirA);
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZnnz((cusparseHandle_t)handle, cuda_dirA, m, n, (const cusparseMatDescr_t)descrA, A, lda, nnzPerRowCol, nnzTotalDevHostPtr);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZnnz_compress(UPTKsparseHandle_t handle, int m, const UPTKsparseMatDescr_t descr, const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA, int *nnzPerRow, int *nnzC, cuDoubleComplex tol)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseZnnz_compress((cusparseHandle_t)handle, m, (const cusparseMatDescr_t)descr, csrSortedValA, csrSortedRowPtrA, nnzPerRow, nnzC, tol);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCsr2cscEx2(UPTKsparseHandle_t handle, int m, int n, int nnz, const void *csrVal, const int *csrRowPtr, const int *csrColInd, void *cscVal, int *cscColPtr, int *cscRowInd, UPTKDataType valType, UPTKsparseAction_t copyValues, UPTKsparseIndexBase_t idxBase, UPTKsparseCsr2CscAlg_t alg, void *buffer)
{
    return cusparseStatusToUPTKsparseStatus(cusparseCsr2cscEx2((cusparseHandle_t)handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscColPtr, cscRowInd, UPTKDataTypeTocudaDataType(valType), UPTKsparseActionTocusparseAction(copyValues), UPTKsparseIndexBaseTocusparseIndexBase(idxBase), UPTKsparseCsr2CscAlgTocusparseCsr2CscAlg(alg), buffer));
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCsr2cscEx2_bufferSize(UPTKsparseHandle_t handle, int m, int n, int nnz, const void *csrVal, const int *csrRowPtr, const int *csrColInd, void *cscVal, int *cscColPtr, int *cscRowInd, UPTKDataType valType, UPTKsparseAction_t copyValues, UPTKsparseIndexBase_t idxBase, UPTKsparseCsr2CscAlg_t alg, size_t *bufferSize)
{
    return cusparseStatusToUPTKsparseStatus(cusparseCsr2cscEx2_bufferSize((cusparseHandle_t)handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscColPtr, cscRowInd, UPTKDataTypeTocudaDataType(valType), UPTKsparseActionTocusparseAction(copyValues), UPTKsparseIndexBaseTocusparseIndexBase(idxBase), UPTKsparseCsr2CscAlgTocusparseCsr2CscAlg(alg), bufferSize));
}

#ifdef UPTK_NOT_SUPPORT
UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseHpruneCsr2csr(UPTKsparseHandle_t handle, int m, int n, int nnzA, const UPTKsparseMatDescr_t descrA, const __half *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, const __half *threshold, const UPTKsparseMatDescr_t descrC, __half *csrSortedValC, const int *csrSortedRowPtrC, int *csrSortedColIndC, void *pBuffer)
{
    Debug();
    return UPTKSPARSE_STATUS_NOT_SUPPORTED;
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseHpruneCsr2csr_bufferSizeExt(UPTKsparseHandle_t handle, int m, int n, int nnzA, const UPTKsparseMatDescr_t descrA, const __half *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, const __half *threshold, const UPTKsparseMatDescr_t descrC, const __half *csrSortedValC, const int *csrSortedRowPtrC, const int *csrSortedColIndC, size_t *pBufferSizeInBytes)
{
    Debug();
    return UPTKSPARSE_STATUS_NOT_SUPPORTED;
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseHpruneCsr2csrByPercentage(UPTKsparseHandle_t handle, int m, int n, int nnzA, const UPTKsparseMatDescr_t descrA, const __half *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage, const UPTKsparseMatDescr_t descrC, __half *csrSortedValC, const int *csrSortedRowPtrC, int *csrSortedColIndC, pruneInfo_t info, void *pBuffer)
{
    Debug();
    return UPTKSPARSE_STATUS_NOT_SUPPORTED;
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseHpruneCsr2csrByPercentage_bufferSizeExt(UPTKsparseHandle_t handle, int m, int n, int nnzA, const UPTKsparseMatDescr_t descrA, const __half *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage, const UPTKsparseMatDescr_t descrC, const __half *csrSortedValC, const int *csrSortedRowPtrC, const int *csrSortedColIndC, pruneInfo_t info, size_t *pBufferSizeInBytes)
{
    Debug();
    return UPTKSPARSE_STATUS_NOT_SUPPORTED;
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseHpruneCsr2csrNnz(UPTKsparseHandle_t handle, int m, int n, int nnzA, const UPTKsparseMatDescr_t descrA, const __half *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, const __half *threshold, const UPTKsparseMatDescr_t descrC, int *csrSortedRowPtrC, int *nnzTotalDevHostPtr, void *pBuffer)
{
    Debug();
    return UPTKSPARSE_STATUS_NOT_SUPPORTED;
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseHpruneCsr2csrNnzByPercentage(UPTKsparseHandle_t handle, int m, int n, int nnzA, const UPTKsparseMatDescr_t descrA, const __half *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, float percentage, const UPTKsparseMatDescr_t descrC, int *csrSortedRowPtrC, int *nnzTotalDevHostPtr, pruneInfo_t info, void *pBuffer)
{
    Debug();
    return UPTKSPARSE_STATUS_NOT_SUPPORTED;
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseHpruneDense2csr(UPTKsparseHandle_t handle, int m, int n, const __half *A, int lda, const __half *threshold, const UPTKsparseMatDescr_t descrC, __half *csrSortedValC, const int *csrSortedRowPtrC, int *csrSortedColIndC, void *pBuffer)
{
    Debug();
    return UPTKSPARSE_STATUS_NOT_SUPPORTED;
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseHpruneDense2csr_bufferSizeExt(UPTKsparseHandle_t handle, int m, int n, const __half *A, int lda, const __half *threshold, const UPTKsparseMatDescr_t descrC, const __half *csrSortedValC, const int *csrSortedRowPtrC, const int *csrSortedColIndC, size_t *pBufferSizeInBytes)
{
    Debug();
    return UPTKSPARSE_STATUS_NOT_SUPPORTED;
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseHpruneDense2csrByPercentage(UPTKsparseHandle_t handle, int m, int n, const __half *A, int lda, float percentage, const UPTKsparseMatDescr_t descrC, __half *csrSortedValC, const int *csrSortedRowPtrC, int *csrSortedColIndC, pruneInfo_t info, void *pBuffer)
{
    Debug();
    return UPTKSPARSE_STATUS_NOT_SUPPORTED;
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseHpruneDense2csrByPercentage_bufferSizeExt(UPTKsparseHandle_t handle, int m, int n, const __half *A, int lda, float percentage, const UPTKsparseMatDescr_t descrC, const __half *csrSortedValC, const int *csrSortedRowPtrC, const int *csrSortedColIndC, pruneInfo_t info, size_t *pBufferSizeInBytes)
{
    Debug();
    return UPTKSPARSE_STATUS_NOT_SUPPORTED;
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseHpruneDense2csrNnz(UPTKsparseHandle_t handle, int m, int n, const __half *A, int lda, const __half *threshold, const UPTKsparseMatDescr_t descrC, int *csrRowPtrC, int *nnzTotalDevHostPtr, void *pBuffer)
{
    Debug();
    return UPTKSPARSE_STATUS_NOT_SUPPORTED;
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseHpruneDense2csrNnzByPercentage(UPTKsparseHandle_t handle, int m, int n, const __half *A, int lda, float percentage, const UPTKsparseMatDescr_t descrC, int *csrRowPtrC, int *nnzTotalDevHostPtr, pruneInfo_t info, void *pBuffer)
{
    Debug();
    return UPTKSPARSE_STATUS_NOT_SUPPORTED;
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSbsrsm2_bufferSizeExt(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, UPTKsparseOperation_t transA, UPTKsparseOperation_t transB, int mb, int n, int nnzb, const UPTKsparseMatDescr_t descrA, float *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info, size_t *pBufferSize)
{
    Debug();
    return UPTKSPARSE_STATUS_NOT_SUPPORTED;
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDbsrsm2_bufferSizeExt(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, UPTKsparseOperation_t transA, UPTKsparseOperation_t transB, int mb, int n, int nnzb, const UPTKsparseMatDescr_t descrA, double *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info, size_t *pBufferSize)
{
    Debug();
    return UPTKSPARSE_STATUS_NOT_SUPPORTED;
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCbsrsm2_bufferSizeExt(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, UPTKsparseOperation_t transA, UPTKsparseOperation_t transB, int mb, int n, int nnzb, const UPTKsparseMatDescr_t descrA, cuComplex *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info, size_t *pBufferSize)
{
    Debug();
    return UPTKSPARSE_STATUS_NOT_SUPPORTED;
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZbsrsm2_bufferSizeExt(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, UPTKsparseOperation_t transA, UPTKsparseOperation_t transB, int mb, int n, int nnzb, const UPTKsparseMatDescr_t descrA, cuDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize, bsrsm2Info_t info, size_t *pBufferSize)
{
    Debug();
    return UPTKSPARSE_STATUS_NOT_SUPPORTED;
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSbsrilu02_bufferSizeExt(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nnzb, const UPTKsparseMatDescr_t descrA, float *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize, bsrilu02Info_t info, size_t *pBufferSize)
{
    Debug();
    return UPTKSPARSE_STATUS_NOT_SUPPORTED;
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDbsrilu02_bufferSizeExt(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nnzb, const UPTKsparseMatDescr_t descrA, double *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize, bsrilu02Info_t info, size_t *pBufferSize)
{
    Debug();
    return UPTKSPARSE_STATUS_NOT_SUPPORTED;
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCbsrilu02_bufferSizeExt(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nnzb, const UPTKsparseMatDescr_t descrA, cuComplex *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize, bsrilu02Info_t info, size_t *pBufferSize)
{
    Debug();
    return UPTKSPARSE_STATUS_NOT_SUPPORTED;
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZbsrilu02_bufferSizeExt(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nnzb, const UPTKsparseMatDescr_t descrA, cuDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize, bsrilu02Info_t info, size_t *pBufferSize)
{
    Debug();
    return UPTKSPARSE_STATUS_NOT_SUPPORTED;
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSbsric02_bufferSizeExt(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nnzb, const UPTKsparseMatDescr_t descrA, float *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize, bsric02Info_t info, size_t *pBufferSize)
{
    Debug();
    return UPTKSPARSE_STATUS_NOT_SUPPORTED;
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDbsric02_bufferSizeExt(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nnzb, const UPTKsparseMatDescr_t descrA, double *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize, bsric02Info_t info, size_t *pBufferSize)
{
    Debug();
    return UPTKSPARSE_STATUS_NOT_SUPPORTED;
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCbsric02_bufferSizeExt(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nnzb, const UPTKsparseMatDescr_t descrA, cuComplex *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize, bsric02Info_t info, size_t *pBufferSize)
{
    Debug();
    return UPTKSPARSE_STATUS_NOT_SUPPORTED;
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZbsric02_bufferSizeExt(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nnzb, const UPTKsparseMatDescr_t descrA, cuDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int blockSize, bsric02Info_t info, size_t *pBufferSize)
{
    Debug();
    return UPTKSPARSE_STATUS_NOT_SUPPORTED;
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSgebsr2gebsc_bufferSizeExt(UPTKsparseHandle_t handle, int mb, int nb, int nnzb, const float *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int rowBlockDim, int colBlockDim, size_t *pBufferSize)
{
    Debug();
    return UPTKSPARSE_STATUS_NOT_SUPPORTED;
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDgebsr2gebsc_bufferSizeExt(UPTKsparseHandle_t handle, int mb, int nb, int nnzb, const double *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int rowBlockDim, int colBlockDim, size_t *pBufferSize)
{
    Debug();
    return UPTKSPARSE_STATUS_NOT_SUPPORTED;
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCgebsr2gebsc_bufferSizeExt(UPTKsparseHandle_t handle, int mb, int nb, int nnzb, const cuComplex *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int rowBlockDim, int colBlockDim, size_t *pBufferSize)
{
    Debug();
    return UPTKSPARSE_STATUS_NOT_SUPPORTED;
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZgebsr2gebsc_bufferSizeExt(UPTKsparseHandle_t handle, int mb, int nb, int nnzb, const cuDoubleComplex *bsrSortedVal, const int *bsrSortedRowPtr, const int *bsrSortedColInd, int rowBlockDim, int colBlockDim, size_t *pBufferSize)
{
    Debug();
    return UPTKSPARSE_STATUS_NOT_SUPPORTED;
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseXgebsr2csr(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nb, const UPTKsparseMatDescr_t descrA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDim, int colBlockDim, const UPTKsparseMatDescr_t descrC, int *csrSortedRowPtrC, int *csrSortedColIndC)
{
    Debug();
    return UPTKSPARSE_STATUS_NOT_SUPPORTED;
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseScsr2gebsr_bufferSizeExt(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int m, int n, const UPTKsparseMatDescr_t descrA, const float *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, int rowBlockDim, int colBlockDim, size_t *pBufferSize)
{
    Debug();
    return UPTKSPARSE_STATUS_NOT_SUPPORTED;
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDcsr2gebsr_bufferSizeExt(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int m, int n, const UPTKsparseMatDescr_t descrA, const double *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, int rowBlockDim, int colBlockDim, size_t *pBufferSize)
{
    Debug();
    return UPTKSPARSE_STATUS_NOT_SUPPORTED;
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCcsr2gebsr_bufferSizeExt(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int m, int n, const UPTKsparseMatDescr_t descrA, const cuComplex *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, int rowBlockDim, int colBlockDim, size_t *pBufferSize)
{
    Debug();
    return UPTKSPARSE_STATUS_NOT_SUPPORTED;
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZcsr2gebsr_bufferSizeExt(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int m, int n, const UPTKsparseMatDescr_t descrA, const cuDoubleComplex *csrSortedValA, const int *csrSortedRowPtrA, const int *csrSortedColIndA, int rowBlockDim, int colBlockDim, size_t *pBufferSize)
{
    Debug();
    return UPTKSPARSE_STATUS_NOT_SUPPORTED;
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSgebsr2gebsr_bufferSizeExt(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nb, int nnzb, const UPTKsparseMatDescr_t descrA, const float *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, int rowBlockDimC, int colBlockDimC, size_t *pBufferSize)
{
    Debug();
    return UPTKSPARSE_STATUS_NOT_SUPPORTED;
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseDgebsr2gebsr_bufferSizeExt(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nb, int nnzb, const UPTKsparseMatDescr_t descrA, const double *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, int rowBlockDimC, int colBlockDimC, size_t *pBufferSize)
{
    Debug();
    return UPTKSPARSE_STATUS_NOT_SUPPORTED;
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCgebsr2gebsr_bufferSizeExt(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nb, int nnzb, const UPTKsparseMatDescr_t descrA, const cuComplex *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, int rowBlockDimC, int colBlockDimC, size_t *pBufferSize)
{
    Debug();
    return UPTKSPARSE_STATUS_NOT_SUPPORTED;
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseZgebsr2gebsr_bufferSizeExt(UPTKsparseHandle_t handle, UPTKsparseDirection_t dirA, int mb, int nb, int nnzb, const UPTKsparseMatDescr_t descrA, const cuDoubleComplex *bsrSortedValA, const int *bsrSortedRowPtrA, const int *bsrSortedColIndA, int rowBlockDimA, int colBlockDimA, int rowBlockDimC, int colBlockDimC, size_t *pBufferSize)
{
    Debug();
    return UPTKSPARSE_STATUS_NOT_SUPPORTED;
}

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseLoggerSetCallback(UPTKsparseLoggerCallback_t callback)
{
    Debug();
    return UPTKSPARSE_STATUS_NOT_SUPPORTED;
}

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseLoggerSetFile(FILE *file)
{
    Debug();
    return UPTKSPARSE_STATUS_NOT_SUPPORTED;
}

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseLoggerOpenFile(const char *logFile)
{
    Debug();
    return UPTKSPARSE_STATUS_NOT_SUPPORTED;
}

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseLoggerSetLevel(int level)
{
    Debug();
    return UPTKSPARSE_STATUS_NOT_SUPPORTED;
}
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseLoggerSetMask(int mask)
{
    Debug();
    return UPTKSPARSE_STATUS_NOT_SUPPORTED;
}

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseLoggerForceDisable(void)
{
    Debug();
    return UPTKSPARSE_STATUS_NOT_SUPPORTED;
}
#endif

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSpMatGetSize(UPTKsparseSpMatDescr_t spMatDescr, int64_t *rows, int64_t *cols, int64_t *nnz)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSpMatGetSize((cusparseSpMatDescr_t)spMatDescr, rows, cols, nnz);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCooSetStridedBatch(UPTKsparseSpMatDescr_t spMatDescr, int batchCount, int64_t batchStride)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCooSetStridedBatch((cusparseSpMatDescr_t)spMatDescr, batchCount, batchStride);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseCsrSetStridedBatch(UPTKsparseSpMatDescr_t spMatDescr, int batchCount, int64_t offsetsBatchStride, int64_t columnsValuesBatchStride)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseCsrSetStridedBatch((cusparseSpMatDescr_t)spMatDescr, batchCount, offsetsBatchStride, columnsValuesBatchStride);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSpMatGetAttribute(UPTKsparseSpMatDescr_t spMatDescr, UPTKsparseSpMatAttribute_t attribute, void *data, size_t dataSize)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSpMatGetAttribute((cusparseSpMatDescr_t)spMatDescr, UPTKsparseSpMatAttributeTocusparseSpMatAttribute(attribute), data, dataSize);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSpMatSetAttribute(UPTKsparseSpMatDescr_t spMatDescr, UPTKsparseSpMatAttribute_t attribute, void *data, size_t dataSize)
{
    cusparseStatus_t cuda_res;
    cuda_res = cusparseSpMatSetAttribute((cusparseSpMatDescr_t)spMatDescr, UPTKsparseSpMatAttributeTocusparseSpMatAttribute(attribute), data, dataSize);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateCsc(UPTKsparseSpMatDescr_t *spMatDescr,
                    int64_t rows,
                    int64_t cols,
                    int64_t nnz,
                    void *cscColOffsets,
                    void *cscRowInd,
                    void *cscValues,
                    UPTKsparseIndexType_t cscColOffsetsType,
                    UPTKsparseIndexType_t cscRowIndType,
                    UPTKsparseIndexBase_t idxBase,
                    UPTKDataType valueType)
{
    return cusparseStatusToUPTKsparseStatus(cusparseCreateCsc((cusparseSpMatDescr_t *)spMatDescr, rows, cols, nnz, cscColOffsets, cscRowInd, cscValues, UPTKsparseIndexTypeTocusparseIndexType(cscColOffsetsType),
                                                              UPTKsparseIndexTypeTocusparseIndexType(cscRowIndType), UPTKsparseIndexBaseTocusparseIndexBase(idxBase), UPTKDataTypeTocudaDataType(valueType)));
}

/*UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCscGet(UPTKsparseSpMatDescr_t spMatDescr,
                 int64_t *rows,
                 int64_t *cols,
                 int64_t *nnz,
                 void **cscColOffsets,
                 void **cscRowInd,
                 void **cscValues,
                 UPTKsparseIndexType_t *cscColOffsetsType,
                 UPTKsparseIndexType_t *cscRowIndType,
                 UPTKsparseIndexBase_t *idxBase,
                 UPTKDataType *valueType)
{
    cusparseStatus_t cuda_res;
    cusparseIndexType_t cudaIdxRowType;
    cusparseIndexType_t cudaIdxColType;
    cusparseIndexBase_t cudaIdxBase;
    cudaDataType cudaType;

    cuda_res = cusparseCscGet((cusparseSpMatDescr_t)spMatDescr, rows,
                                   cols, nnz, (void **)cscColOffsets,
                                   (void **)cscRowInd, (void **)cscValues,
                                   &cudaIdxRowType, &cudaIdxColType, &cudaIdxBase, &cudaType);
    *cscColOffsetsType = cusparseIndexTypeToUPTKsparseIndexType(cudaIdxRowType);
    *cscRowIndType = cusparseIndexTypeToUPTKsparseIndexType(cudaIdxColType);
    *idxBase = cusparseIndexBaseToUPTKsparseIndexBase(cudaIdxBase);
    *valueType = cudaDataTypeToUPTKDataType(cudaType);

    return cusparseStatusToUPTKsparseStatus(cuda_res);
}*/

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCsrSetPointers(UPTKsparseSpMatDescr_t spMatDescr,
                         void *csrRowOffsets,
                         void *csrColInd,
                         void *csrValues)
{
    return cusparseStatusToUPTKsparseStatus(cusparseCsrSetPointers((cusparseSpMatDescr_t)spMatDescr, csrRowOffsets, csrColInd, csrValues));
}

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCscSetPointers(UPTKsparseSpMatDescr_t spMatDescr,
                         void *cscColOffsets,
                         void *cscRowInd,
                         void *cscValues)
{

    return cusparseStatusToUPTKsparseStatus(cusparseCscSetPointers((cusparseSpMatDescr_t)spMatDescr, cscColOffsets, cscRowInd, cscValues));
}

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCooSetPointers(UPTKsparseSpMatDescr_t spMatDescr,
                         void *cooRows,
                         void *cooColumns,
                         void *cooValues)
{

    return cusparseStatusToUPTKsparseStatus(cusparseCooSetPointers((cusparseSpMatDescr_t)spMatDescr, cooRows, cooColumns, cooValues));
}

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateBlockedEll(UPTKsparseSpMatDescr_t *spMatDescr,
                           int64_t rows,
                           int64_t cols,
                           int64_t ellBlockSize,
                           int64_t ellCols,
                           void *ellColInd,
                           void *ellValue,
                           UPTKsparseIndexType_t ellIdxType,
                           UPTKsparseIndexBase_t idxBase,
                           UPTKDataType valueType)
{
    return cusparseStatusToUPTKsparseStatus(cusparseCreateBlockedEll((cusparseSpMatDescr_t *)spMatDescr, rows, cols, ellBlockSize, ellCols, ellColInd, ellValue,
                                                                     UPTKsparseIndexTypeTocusparseIndexType(ellIdxType), UPTKsparseIndexBaseTocusparseIndexBase(idxBase), UPTKDataTypeTocudaDataType(valueType)));
}

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseBlockedEllGet(UPTKsparseSpMatDescr_t spMatDescr,
                        int64_t *rows,
                        int64_t *cols,
                        int64_t *ellBlockSize,
                        int64_t *ellCols,
                        void **ellColInd,
                        void **ellValue,
                        UPTKsparseIndexType_t *ellIdxType,
                        UPTKsparseIndexBase_t *idxBase,
                        UPTKDataType *valueType)
{
    cusparseIndexType_t cuda_idxType;
    cusparseIndexBase_t cuda_idxBase;
    cudaDataType cuda_valueType;
    cusparseStatus_t cuda_res;
    cuda_res = cusparseBlockedEllGet((const cusparseSpMatDescr_t)spMatDescr, rows, cols, ellBlockSize, ellCols, ellColInd, ellValue, &cuda_idxType, &cuda_idxBase, &cuda_valueType);
    *ellIdxType = cusparseIndexTypeToUPTKsparseIndexType(cuda_idxType);
    *idxBase = cusparseIndexBaseToUPTKsparseIndexBase(cuda_idxBase);
    *valueType = cudaDataTypeToUPTKDataType(cuda_valueType);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseAxpby(UPTKsparseHandle_t handle,
                const void *alpha,
                UPTKsparseSpVecDescr_t vecX,
                const void *beta,
                UPTKsparseDnVecDescr_t vecY)
{
    return cusparseStatusToUPTKsparseStatus(cusparseAxpby((cusparseHandle_t)handle, alpha, (cusparseSpVecDescr_t)vecX, beta, (cusparseDnVecDescr_t)vecY));
}

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseGather(UPTKsparseHandle_t handle,
                 UPTKsparseDnVecDescr_t vecY,
                 UPTKsparseSpVecDescr_t vecX)
{
    return cusparseStatusToUPTKsparseStatus(cusparseGather((cusparseHandle_t)handle, (cusparseDnVecDescr_t)vecY, (cusparseSpVecDescr_t)vecX));
}

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseScatter(UPTKsparseHandle_t handle,
                  UPTKsparseSpVecDescr_t vecX,
                  UPTKsparseDnVecDescr_t vecY)
{
    return cusparseStatusToUPTKsparseStatus(cusparseScatter((cusparseHandle_t)handle, (cusparseSpVecDescr_t)vecX, (cusparseDnVecDescr_t)vecY));
}

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseRot(UPTKsparseHandle_t handle,
              const void *c_coeff,
              const void *s_coeff,
              UPTKsparseSpVecDescr_t vecX,
              UPTKsparseDnVecDescr_t vecY)
{
    return cusparseStatusToUPTKsparseStatus(cusparseRot((cusparseHandle_t)handle, c_coeff, s_coeff, (cusparseSpVecDescr_t)vecX, (cusparseDnVecDescr_t)vecY));
}

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSparseToDense_bufferSize(UPTKsparseHandle_t handle,
                                   UPTKsparseSpMatDescr_t matA,
                                   UPTKsparseDnMatDescr_t matB,
                                   UPTKsparseSparseToDenseAlg_t alg,
                                   size_t *bufferSize)
{
    return cusparseStatusToUPTKsparseStatus(cusparseSparseToDense_bufferSize((cusparseHandle_t)handle, (cusparseSpMatDescr_t)matA, (cusparseDnMatDescr_t)matB, UPTKsparseSparseToDenseAlgTocusparseSparseToDenseAlg(alg), bufferSize));
}

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSparseToDense(UPTKsparseHandle_t handle,
                        UPTKsparseSpMatDescr_t matA,
                        UPTKsparseDnMatDescr_t matB,
                        UPTKsparseSparseToDenseAlg_t alg,
                        void *externalBuffer)
{
    return cusparseStatusToUPTKsparseStatus(cusparseSparseToDense((cusparseHandle_t)handle, (cusparseSpMatDescr_t)matA, (cusparseDnMatDescr_t)matB, UPTKsparseSparseToDenseAlgTocusparseSparseToDenseAlg(alg), externalBuffer));
}

// #############################################################################
// # DENSE TO SPARSE
// #############################################################################

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDenseToSparse_bufferSize(UPTKsparseHandle_t handle,
                                   UPTKsparseDnMatDescr_t matA,
                                   UPTKsparseSpMatDescr_t matB,
                                   UPTKsparseDenseToSparseAlg_t alg,
                                   size_t *bufferSize)
{
    return cusparseStatusToUPTKsparseStatus(cusparseDenseToSparse_bufferSize((cusparseHandle_t)handle, (cusparseDnMatDescr_t)matA, (cusparseSpMatDescr_t)matB, UPTKsparseDenseToSparseAlgTocusparseDenseToSparseAlg(alg), bufferSize));
}

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDenseToSparse_analysis(UPTKsparseHandle_t handle,
                                 UPTKsparseDnMatDescr_t matA,
                                 UPTKsparseSpMatDescr_t matB,
                                 UPTKsparseDenseToSparseAlg_t alg,
                                 void *externalBuffer)
{
    return cusparseStatusToUPTKsparseStatus(cusparseDenseToSparse_analysis((cusparseHandle_t)handle, (cusparseDnMatDescr_t)matA, (cusparseSpMatDescr_t)matB, UPTKsparseDenseToSparseAlgTocusparseDenseToSparseAlg(alg), externalBuffer));
}

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseDenseToSparse_convert(UPTKsparseHandle_t handle,
                                UPTKsparseDnMatDescr_t matA,
                                UPTKsparseSpMatDescr_t matB,
                                UPTKsparseDenseToSparseAlg_t alg,
                                void *externalBuffer)
{
    return cusparseStatusToUPTKsparseStatus(cusparseDenseToSparse_convert((cusparseHandle_t)handle, (cusparseDnMatDescr_t)matA, (cusparseSpMatDescr_t)matB, UPTKsparseDenseToSparseAlgTocusparseDenseToSparseAlg(alg), externalBuffer));
}

// #############################################################################
// # SPARSE TRIANGULAR VECTOR SOLVE
// #############################################################################

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpSV_createDescr(UPTKsparseSpSVDescr_t *descr)
{
    return cusparseStatusToUPTKsparseStatus(cusparseSpSV_createDescr((cusparseSpSVDescr_t *)descr));
}

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpSV_destroyDescr(UPTKsparseSpSVDescr_t descr)
{
    return cusparseStatusToUPTKsparseStatus(cusparseSpSV_destroyDescr((cusparseSpSVDescr_t)descr));
}

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpSV_bufferSize(UPTKsparseHandle_t handle,
                          UPTKsparseOperation_t opA,
                          const void *alpha,
                          UPTKsparseSpMatDescr_t matA,
                          UPTKsparseDnVecDescr_t vecX,
                          UPTKsparseDnVecDescr_t vecY,
                          UPTKDataType computeType,
                          UPTKsparseSpSVAlg_t alg,
                          UPTKsparseSpSVDescr_t spsvDescr,
                          size_t *bufferSize)
{
    return cusparseStatusToUPTKsparseStatus(cusparseSpSV_bufferSize((cusparseHandle_t)handle, UPTKsparseOperationTocusparseOperation(opA), alpha, (cusparseSpMatDescr_t)matA, (cusparseDnVecDescr_t)vecX,
                                                                    (cusparseDnVecDescr_t)vecY, UPTKDataTypeTocudaDataType(computeType), UPTKsparseSpSVAlgTocusparseSpSVAlg(alg), (cusparseSpSVDescr_t)spsvDescr, bufferSize));
}

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpSV_analysis(UPTKsparseHandle_t handle,
                        UPTKsparseOperation_t opA,
                        const void *alpha,
                        UPTKsparseSpMatDescr_t matA,
                        UPTKsparseDnVecDescr_t vecX,
                        UPTKsparseDnVecDescr_t vecY,
                        UPTKDataType computeType,
                        UPTKsparseSpSVAlg_t alg,
                        UPTKsparseSpSVDescr_t spsvDescr,
                        void *externalBuffer)
{
    return cusparseStatusToUPTKsparseStatus(cusparseSpSV_analysis((cusparseHandle_t)handle, UPTKsparseOperationTocusparseOperation(opA), alpha, (cusparseSpMatDescr_t)matA, (cusparseDnVecDescr_t)vecX,
                                                                  (cusparseDnVecDescr_t)vecY, UPTKDataTypeTocudaDataType(computeType), UPTKsparseSpSVAlgTocusparseSpSVAlg(alg), (cusparseSpSVDescr_t)spsvDescr, externalBuffer));
}
// 参数不一致
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpSV_solve(UPTKsparseHandle_t handle,
                     UPTKsparseOperation_t opA,
                     const void *alpha,
                     UPTKsparseSpMatDescr_t matA,
                     UPTKsparseDnVecDescr_t vecX,
                     UPTKsparseDnVecDescr_t vecY,
                     UPTKDataType computeType,
                     UPTKsparseSpSVAlg_t alg,
                     UPTKsparseSpSVDescr_t spsvDescr)
{
    return cusparseStatusToUPTKsparseStatus(cusparseSpSV_solve((cusparseHandle_t)handle, UPTKsparseOperationTocusparseOperation(opA), alpha, (const cusparseSpMatDescr_t)matA, (const cusparseDnVecDescr_t)vecX,
                                                               (cusparseDnVecDescr_t)vecY, UPTKDataTypeTocudaDataType(computeType), UPTKsparseSpSVAlgTocusparseSpSVAlg(alg), (cusparseSpSVDescr_t)spsvDescr));
}

// #############################################################################
// # SPARSE TRIANGULAR MATRIX SOLVE
// #############################################################################

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpSM_createDescr(UPTKsparseSpSMDescr_t *descr)
{
    return cusparseStatusToUPTKsparseStatus(cusparseSpSM_createDescr((cusparseSpSMDescr_t *)descr));
}

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpSM_destroyDescr(UPTKsparseSpSMDescr_t descr)
{
    return cusparseStatusToUPTKsparseStatus(cusparseSpSM_destroyDescr((cusparseSpSMDescr_t)descr));
}

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpSM_bufferSize(UPTKsparseHandle_t handle,
                          UPTKsparseOperation_t opA,
                          UPTKsparseOperation_t opB,
                          const void *alpha,
                          UPTKsparseSpMatDescr_t matA,
                          UPTKsparseDnMatDescr_t matB,
                          UPTKsparseDnMatDescr_t matC,
                          UPTKDataType computeType,
                          UPTKsparseSpSMAlg_t alg,
                          UPTKsparseSpSMDescr_t spsmDescr,
                          size_t *bufferSize)
{
    return cusparseStatusToUPTKsparseStatus(cusparseSpSM_bufferSize((cusparseHandle_t)handle, UPTKsparseOperationTocusparseOperation(opA), UPTKsparseOperationTocusparseOperation(opB), alpha,
                                                                    (cusparseSpMatDescr_t)matA, (cusparseDnMatDescr_t)matB, (cusparseDnMatDescr_t)matC, UPTKDataTypeTocudaDataType(computeType), UPTKsparseSpSMAlgTocusparseSpSMAlg(alg), (cusparseSpSMDescr_t)spsmDescr, bufferSize));
}

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpSM_analysis(UPTKsparseHandle_t handle,
                        UPTKsparseOperation_t opA,
                        UPTKsparseOperation_t opB,
                        const void *alpha,
                        UPTKsparseSpMatDescr_t matA,
                        UPTKsparseDnMatDescr_t matB,
                        UPTKsparseDnMatDescr_t matC,
                        UPTKDataType computeType,
                        UPTKsparseSpSMAlg_t alg,
                        UPTKsparseSpSMDescr_t spsmDescr,
                        void *externalBuffer)
{
    return cusparseStatusToUPTKsparseStatus(cusparseSpSM_analysis((cusparseHandle_t)handle, UPTKsparseOperationTocusparseOperation(opA), UPTKsparseOperationTocusparseOperation(opB), alpha,
                                                                  (cusparseSpMatDescr_t)matA, (cusparseDnMatDescr_t)matB, (cusparseDnMatDescr_t)matC, UPTKDataTypeTocudaDataType(computeType), UPTKsparseSpSMAlgTocusparseSpSMAlg(alg), (cusparseSpSMDescr_t)spsmDescr, externalBuffer));
}

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpSM_solve(UPTKsparseHandle_t handle,
                     UPTKsparseOperation_t opA,
                     UPTKsparseOperation_t opB,
                     const void *alpha,
                     UPTKsparseSpMatDescr_t matA,
                     UPTKsparseDnMatDescr_t matB,
                     UPTKsparseDnMatDescr_t matC,
                     UPTKDataType computeType,
                     UPTKsparseSpSMAlg_t alg,
                     UPTKsparseSpSMDescr_t spsmDescr)
{
    return cusparseStatusToUPTKsparseStatus(cusparseSpSM_solve((cusparseHandle_t)handle, UPTKsparseOperationTocusparseOperation(opA), UPTKsparseOperationTocusparseOperation(opB), alpha, (cusparseSpMatDescr_t)matA,
                                                               (cusparseDnMatDescr_t)matB, (const cusparseDnMatDescr_t)matC, UPTKDataTypeTocudaDataType(computeType), UPTKsparseSpSMAlgTocusparseSpSMAlg(alg), (cusparseSpSMDescr_t)spsmDescr));
}

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpMM_preprocess(UPTKsparseHandle_t handle,
                          UPTKsparseOperation_t opA,
                          UPTKsparseOperation_t opB,
                          const void *alpha,
                          UPTKsparseSpMatDescr_t matA,
                          UPTKsparseDnMatDescr_t matB,
                          const void *beta,
                          UPTKsparseDnMatDescr_t matC,
                          UPTKDataType computeType,
                          UPTKsparseSpMMAlg_t alg,
                          void *externalBuffer)
{
    return cusparseStatusToUPTKsparseStatus(cusparseSpMM_preprocess((cusparseHandle_t)handle,
                                                                    UPTKsparseOperationTocusparseOperation(opA),
                                                                    UPTKsparseOperationTocusparseOperation(opB), alpha,
                                                                    (cusparseSpMatDescr_t)matA,
                                                                    (cusparseDnMatDescr_t)matB, beta,
                                                                    (cusparseDnMatDescr_t)matC,
                                                                    UPTKDataTypeTocudaDataType(computeType),
                                                                    UPTKsparseSpMMAlgTocusparseSpMMAlg(alg),
                                                                    externalBuffer));
}

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpGEMM_createDescr(UPTKsparseSpGEMMDescr_t *descr)
{
    return cusparseStatusToUPTKsparseStatus(cusparseSpGEMM_createDescr((cusparseSpGEMMDescr_t *)descr));
}

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpGEMM_destroyDescr(UPTKsparseSpGEMMDescr_t descr)
{
    return cusparseStatusToUPTKsparseStatus(cusparseSpGEMM_destroyDescr((cusparseSpGEMMDescr_t)descr));
}

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpGEMM_workEstimation(UPTKsparseHandle_t handle,
                                UPTKsparseOperation_t opA,
                                UPTKsparseOperation_t opB,
                                const void *alpha,
                                UPTKsparseSpMatDescr_t matA,
                                UPTKsparseSpMatDescr_t matB,
                                const void *beta,
                                UPTKsparseSpMatDescr_t matC,
                                UPTKDataType computeType,
                                UPTKsparseSpGEMMAlg_t alg,
                                UPTKsparseSpGEMMDescr_t spgemmDescr,
                                size_t *bufferSize1,
                                void *externalBuffer1)
{
    return cusparseStatusToUPTKsparseStatus(cusparseSpGEMM_workEstimation((cusparseHandle_t)handle, UPTKsparseOperationTocusparseOperation(opA), UPTKsparseOperationTocusparseOperation(opB), alpha,
                                                                          (cusparseSpMatDescr_t)matA, (cusparseSpMatDescr_t)matB, beta, (cusparseSpMatDescr_t)matC, UPTKDataTypeTocudaDataType(computeType), UPTKsparseSpGEMMAlgTocusparseSpGEMMAlg(alg), (cusparseSpGEMMDescr_t)spgemmDescr, bufferSize1, externalBuffer1));
}

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpGEMM_compute(UPTKsparseHandle_t handle,
                         UPTKsparseOperation_t opA,
                         UPTKsparseOperation_t opB,
                         const void *alpha,
                         UPTKsparseSpMatDescr_t matA,
                         UPTKsparseSpMatDescr_t matB,
                         const void *beta,
                         UPTKsparseSpMatDescr_t matC,
                         UPTKDataType computeType,
                         UPTKsparseSpGEMMAlg_t alg,
                         UPTKsparseSpGEMMDescr_t spgemmDescr,
                         size_t *bufferSize2,
                         void *externalBuffer2)
{
    return cusparseStatusToUPTKsparseStatus(cusparseSpGEMM_compute((cusparseHandle_t)handle, UPTKsparseOperationTocusparseOperation(opA), UPTKsparseOperationTocusparseOperation(opB), alpha,
                                                                   (cusparseSpMatDescr_t)matA, (cusparseSpMatDescr_t)matB, beta, (cusparseSpMatDescr_t)matC, UPTKDataTypeTocudaDataType(computeType), UPTKsparseSpGEMMAlgTocusparseSpGEMMAlg(alg), (cusparseSpGEMMDescr_t)spgemmDescr, bufferSize2, externalBuffer2));
}

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpGEMM_copy(UPTKsparseHandle_t handle,
                      UPTKsparseOperation_t opA,
                      UPTKsparseOperation_t opB,
                      const void *alpha,
                      UPTKsparseSpMatDescr_t matA,
                      UPTKsparseSpMatDescr_t matB,
                      const void *beta,
                      UPTKsparseSpMatDescr_t matC,
                      UPTKDataType computeType,
                      UPTKsparseSpGEMMAlg_t alg,
                      UPTKsparseSpGEMMDescr_t spgemmDescr)
{
    return cusparseStatusToUPTKsparseStatus(cusparseSpGEMM_copy((cusparseHandle_t)handle, UPTKsparseOperationTocusparseOperation(opA), UPTKsparseOperationTocusparseOperation(opB), alpha,
                                                                (cusparseSpMatDescr_t)matA, (cusparseSpMatDescr_t)matB, beta, (cusparseSpMatDescr_t)matC, UPTKDataTypeTocudaDataType(computeType), UPTKsparseSpGEMMAlgTocusparseSpGEMMAlg(alg), (cusparseSpGEMMDescr_t)spgemmDescr));
}

// #############################################################################
// # SPARSE MATRIX - SPARSE MATRIX MULTIPLICATION (SpGEMM) STRUCTURE REUSE
// #############################################################################

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpGEMMreuse_workEstimation(UPTKsparseHandle_t handle,
                                     UPTKsparseOperation_t opA,
                                     UPTKsparseOperation_t opB,
                                     UPTKsparseSpMatDescr_t matA,
                                     UPTKsparseSpMatDescr_t matB,
                                     UPTKsparseSpMatDescr_t matC,
                                     UPTKsparseSpGEMMAlg_t alg,
                                     UPTKsparseSpGEMMDescr_t spgemmDescr,
                                     size_t *bufferSize1,
                                     void *externalBuffer1)
{
    return cusparseStatusToUPTKsparseStatus(cusparseSpGEMMreuse_workEstimation((cusparseHandle_t)handle, UPTKsparseOperationTocusparseOperation(opA), UPTKsparseOperationTocusparseOperation(opB),
                                                                               (cusparseSpMatDescr_t)matA, (cusparseSpMatDescr_t)matB, (cusparseSpMatDescr_t)matC, UPTKsparseSpGEMMAlgTocusparseSpGEMMAlg(alg), (cusparseSpGEMMDescr_t)spgemmDescr, bufferSize1, externalBuffer1));
}

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpGEMMreuse_nnz(UPTKsparseHandle_t handle,
                          UPTKsparseOperation_t opA,
                          UPTKsparseOperation_t opB,
                          UPTKsparseSpMatDescr_t matA,
                          UPTKsparseSpMatDescr_t matB,
                          UPTKsparseSpMatDescr_t matC,
                          UPTKsparseSpGEMMAlg_t alg,
                          UPTKsparseSpGEMMDescr_t spgemmDescr,
                          size_t *bufferSize2,
                          void *externalBuffer2,
                          size_t *bufferSize3,
                          void *externalBuffer3,
                          size_t *bufferSize4,
                          void *externalBuffer4)
{
    return cusparseStatusToUPTKsparseStatus(cusparseSpGEMMreuse_nnz((cusparseHandle_t)handle, UPTKsparseOperationTocusparseOperation(opA), UPTKsparseOperationTocusparseOperation(opB),
                                                                    (cusparseSpMatDescr_t)matA, (cusparseSpMatDescr_t)matB, (cusparseSpMatDescr_t)matC, UPTKsparseSpGEMMAlgTocusparseSpGEMMAlg(alg), (cusparseSpGEMMDescr_t)spgemmDescr,
                                                                    bufferSize2, externalBuffer2, bufferSize3, externalBuffer3, bufferSize4, externalBuffer4));
}

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpGEMMreuse_copy(UPTKsparseHandle_t handle,
                           UPTKsparseOperation_t opA,
                           UPTKsparseOperation_t opB,
                           UPTKsparseSpMatDescr_t matA,
                           UPTKsparseSpMatDescr_t matB,
                           UPTKsparseSpMatDescr_t matC,
                           UPTKsparseSpGEMMAlg_t alg,
                           UPTKsparseSpGEMMDescr_t spgemmDescr,
                           size_t *bufferSize5,
                           void *externalBuffer5)
{
    return cusparseStatusToUPTKsparseStatus(cusparseSpGEMMreuse_copy((cusparseHandle_t)handle, UPTKsparseOperationTocusparseOperation(opA), UPTKsparseOperationTocusparseOperation(opB),
                                                                     (cusparseSpMatDescr_t)matA, (cusparseSpMatDescr_t)matB, (cusparseSpMatDescr_t)matC, UPTKsparseSpGEMMAlgTocusparseSpGEMMAlg(alg), (cusparseSpGEMMDescr_t)spgemmDescr, bufferSize5, externalBuffer5));
}

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpGEMMreuse_compute(UPTKsparseHandle_t handle,
                              UPTKsparseOperation_t opA,
                              UPTKsparseOperation_t opB,
                              const void *alpha,
                              UPTKsparseSpMatDescr_t matA,
                              UPTKsparseSpMatDescr_t matB,
                              const void *beta,
                              UPTKsparseSpMatDescr_t matC,
                              UPTKDataType computeType,
                              UPTKsparseSpGEMMAlg_t alg,
                              UPTKsparseSpGEMMDescr_t spgemmDescr)
{
    return cusparseStatusToUPTKsparseStatus(cusparseSpGEMMreuse_compute((cusparseHandle_t)handle, UPTKsparseOperationTocusparseOperation(opA), UPTKsparseOperationTocusparseOperation(opB), alpha,
                                                                        (cusparseSpMatDescr_t)matA, (cusparseSpMatDescr_t)matB, beta, (cusparseSpMatDescr_t)matC, UPTKDataTypeTocudaDataType(computeType), UPTKsparseSpGEMMAlgTocusparseSpGEMMAlg(alg), (cusparseSpGEMMDescr_t)spgemmDescr));
}

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSDDMM_bufferSize(UPTKsparseHandle_t handle,
                           UPTKsparseOperation_t opA,
                           UPTKsparseOperation_t opB,
                           const void *alpha,
                           UPTKsparseDnMatDescr_t matA,
                           UPTKsparseDnMatDescr_t matB,
                           const void *beta,
                           UPTKsparseSpMatDescr_t matC,
                           UPTKDataType computeType,
                           UPTKsparseSDDMMAlg_t alg,
                           size_t *bufferSize)
{
    return cusparseStatusToUPTKsparseStatus(cusparseSDDMM_bufferSize((cusparseHandle_t)handle, UPTKsparseOperationTocusparseOperation(opA), UPTKsparseOperationTocusparseOperation(opB), alpha,
                                                                     (cusparseDnMatDescr_t)matA, (cusparseDnMatDescr_t)matB, beta, (cusparseSpMatDescr_t)matC, UPTKDataTypeTocudaDataType(computeType), UPTKsparseSDDMMAlgTocusparseSDDMMAlg(alg), bufferSize));
}

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSDDMM_preprocess(UPTKsparseHandle_t handle,
                           UPTKsparseOperation_t opA,
                           UPTKsparseOperation_t opB,
                           const void *alpha,
                           UPTKsparseDnMatDescr_t matA,
                           UPTKsparseDnMatDescr_t matB,
                           const void *beta,
                           UPTKsparseSpMatDescr_t matC,
                           UPTKDataType computeType,
                           UPTKsparseSDDMMAlg_t alg,
                           void *externalBuffer)
{
    return cusparseStatusToUPTKsparseStatus(cusparseSDDMM_preprocess((cusparseHandle_t)handle, UPTKsparseOperationTocusparseOperation(opA), UPTKsparseOperationTocusparseOperation(opB), alpha,
                                                                     (cusparseDnMatDescr_t)matA, (cusparseDnMatDescr_t)matB, beta, (cusparseSpMatDescr_t)matC, UPTKDataTypeTocudaDataType(computeType), UPTKsparseSDDMMAlgTocusparseSDDMMAlg(alg), externalBuffer));
}

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSDDMM(UPTKsparseHandle_t handle,
                UPTKsparseOperation_t opA,
                UPTKsparseOperation_t opB,
                const void *alpha,
                UPTKsparseDnMatDescr_t matA,
                UPTKsparseDnMatDescr_t matB,
                const void *beta,
                UPTKsparseSpMatDescr_t matC,
                UPTKDataType computeType,
                UPTKsparseSDDMMAlg_t alg,
                void *externalBuffer)
{
    return cusparseStatusToUPTKsparseStatus(cusparseSDDMM((cusparseHandle_t)handle, UPTKsparseOperationTocusparseOperation(opA), UPTKsparseOperationTocusparseOperation(opB), alpha,
                                                          (cusparseDnMatDescr_t)matA, (cusparseDnMatDescr_t)matB, beta, (cusparseSpMatDescr_t)matC, UPTKDataTypeTocudaDataType(computeType), UPTKsparseSDDMMAlgTocusparseSDDMMAlg(alg), externalBuffer));
}

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpMMOp_createPlan(UPTKsparseHandle_t handle,
                            UPTKsparseSpMMOpPlan_t *plan,
                            UPTKsparseOperation_t opA,
                            UPTKsparseOperation_t opB,
                            UPTKsparseSpMatDescr_t matA,
                            UPTKsparseDnMatDescr_t matB,
                            UPTKsparseDnMatDescr_t matC,
                            UPTKDataType computeType,
                            UPTKsparseSpMMOpAlg_t alg,
                            const void *addOperationNvvmBuffer,
                            size_t addOperationBufferSize,
                            const void *mulOperationNvvmBuffer,
                            size_t mulOperationBufferSize,
                            const void *epilogueNvvmBuffer,
                            size_t epilogueBufferSize,
                            size_t *SpMMWorkspaceSize)
{
    cusparseStatus_t cuda_res;
    cusparseOperation_t cudaOpA = UPTKsparseOperationTocusparseOperation(opA);
    cusparseOperation_t cudaOpB = UPTKsparseOperationTocusparseOperation(opB);
    cudaDataType cudaType = UPTKDataTypeTocudaDataType(computeType);
    cusparseSpMMOpAlg_t cudaSpMMAlg = UPTKsparseSpMMOpAlgTocusparseSpMMOpAlg(alg);

    cuda_res = cusparseSpMMOp_createPlan((cusparseHandle_t)handle, (cusparseSpMMOpPlan_t *)plan, cudaOpA, cudaOpB,
                                         (cusparseSpMatDescr_t)matA, (cusparseDnMatDescr_t)matB,
                                         (cusparseDnMatDescr_t)matC, cudaType, cudaSpMMAlg, addOperationNvvmBuffer,
                                         addOperationBufferSize, mulOperationNvvmBuffer, mulOperationBufferSize,
                                         epilogueNvvmBuffer, epilogueBufferSize, SpMMWorkspaceSize);

    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSpMMOp(UPTKsparseSpMMOpPlan_t plan, void *externalBuffer)
{
    return cusparseStatusToUPTKsparseStatus(cusparseSpMMOp((cusparseSpMMOpPlan_t)plan, externalBuffer));
}

UPTKsparseStatus_t UPTKSPARSEAPI UPTKsparseSpMMOp_destroyPlan(UPTKsparseSpMMOpPlan_t plan)
{
    return cusparseStatusToUPTKsparseStatus(cusparseSpMMOp_destroyPlan((cusparseSpMMOpPlan_t)plan));
}

/*UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpGEMM_getNumProducts(UPTKsparseSpGEMMDescr_t spgemmDescr,
                                int64_t *num_prods)
{
    return cusparseStatusToUPTKsparseStatus(UPTKsparseSpGEMM_getNumProducts((cusparseSpGEMMDescr_t)spgemmDescr, num_prods));
}

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpGEMM_estimateMemory(UPTKsparseHandle_t handle,
                                UPTKsparseOperation_t opA,
                                UPTKsparseOperation_t opB,
                                const void *alpha,
                                UPTKsparseSpMatDescr_t matA,
                                UPTKsparseSpMatDescr_t matB,
                                const void *beta,
                                UPTKsparseSpMatDescr_t matC,
                                UPTKDataType computeType,
                                UPTKsparseSpGEMMAlg_t alg,
                                UPTKsparseSpGEMMDescr_t spgemmDescr,
                                float chunk_fraction,
                                size_t *bufferSize3,
                                void *externalBuffer3,
                                size_t *bufferSize2)
{
    cusparseStatus_t cuda_res;
    cusparseOperation_t cudaOpA = UPTKsparseOperationTocusparseOperation(opA);
    cusparseOperation_t cudaOpB = UPTKsparseOperationTocusparseOperation(opB);
    cudaDataType cudaType = UPTKDataTypeTocudaDataType(computeType);
    cusparseSpGEMMAlg_t cudaSpMMAlg = UPTKsparseSpGEMMAlgTocusparseSpGEMMAlg(alg);

    cuda_res = cusparseSpGEMM_estimateMemory((cusparseHandle_t)handle,
                                             cudaOpA,
                                             cudaOpB,
                                             alpha,
                                             (cusparseSpMatDescr_t)matA,
                                             (cusparseSpMatDescr_t)matB,
                                             beta,
                                             (cusparseSpMatDescr_t)matC,
                                             cudaType,
                                             cudaSpMMAlg,
                                             (cusparseSpGEMMDescr_t)spgemmDescr,
                                             chunk_fraction,
                                             bufferSize3,
                                             externalBuffer3,
                                             bufferSize2);

    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseBsrSetStridedBatch(UPTKsparseSpMatDescr_t spMatDescr,
                             int batchCount,
                             int64_t offsetsBatchStride,
                             int64_t columnsBatchStride,
                             int64_t ValuesBatchStride)
{
    return cusparseStatusToUPTKsparseStatus(cusparseBsrSetStridedBatch((cusparseSpMatDescr_t)spMatDescr, batchCount, offsetsBatchStride, columnsBatchStride, ValuesBatchStride));
}
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseConstBlockedEllGet(UPTKsparseSpMatDescr_t spMatDescr,
                             int64_t *rows,
                             int64_t *cols,
                             int64_t *ellBlockSize,
                             int64_t *ellCols,
                             const void **ellColInd,
                             const void **ellValue,
                             UPTKsparseIndexType_t *ellIdxType,
                             UPTKsparseIndexBase_t *idxBase,
                             UPTKDataType *valueType)
{
    cusparseStatus_t cuda_res;
    cusparseIndexType_t cudaIdxType;
    cusparseIndexBase_t cudaIdxBase;
    cudaDataType cudaType;

    cuda_res = cusparseConstBlockedEllGet((cusparseSpMatDescr_t)spMatDescr, rows,
                                          cols, ellBlockSize, ellCols, ellColInd, ellValue,
                                          &cudaIdxType,
                                          &cudaIdxBase,
                                          &cudaType);
    *ellIdxType = cusparseIndexTypeToUPTKsparseIndexType(cudaIdxType);
    *idxBase = cusparseIndexBaseToUPTKsparseIndexBase(cudaIdxBase);
    *valueType = cudaDataTypeToUPTKDataType(cudaType);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseConstCooGet(UPTKsparseSpMatDescr_t spMatDescr,
                      int64_t *rows,
                      int64_t *cols,
                      int64_t *nnz,
                      const void **cooRowInd, // COO row indices
                      const void **cooColInd, // COO column indices
                      const void **cooValues, // COO values
                      UPTKsparseIndexType_t *idxType,
                      UPTKsparseIndexBase_t *idxBase,
                      UPTKDataType *valueType)
{
    cusparseStatus_t cuda_res;
    cusparseIndexType_t cudaIdxType;
    cusparseIndexBase_t cudaIdxBase;
    cudaDataType cudaType;

    cuda_res = cusparseConstCooGet((cusparseSpMatDescr_t)spMatDescr, rows,
                                   cols, nnz, cooRowInd, cooColInd, cooValues,
                                   &cudaIdxType,
                                   &cudaIdxBase,
                                   &cudaType);
    *idxType = cusparseIndexTypeToUPTKsparseIndexType(cudaIdxType);
    *idxBase = cusparseIndexBaseToUPTKsparseIndexBase(cudaIdxBase);
    *valueType = cudaDataTypeToUPTKDataType(cudaType);
    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseConstCscGet(UPTKsparseSpMatDescr_t spMatDescr,
                      int64_t *rows,
                      int64_t *cols,
                      int64_t *nnz,
                      const void **cscColOffsets,
                      const void **cscRowInd,
                      const void **cscValues,
                      UPTKsparseIndexType_t *cscColOffsetsType,
                      UPTKsparseIndexType_t *cscRowIndType,
                      UPTKsparseIndexBase_t *idxBase,
                      UPTKDataType *valueType)
{
    cusparseStatus_t cuda_res;
    cusparseIndexType_t cudaIdxRowType;
    cusparseIndexType_t cudaIdxColType;
    cusparseIndexBase_t cudaIdxBase;
    cudaDataType cudaType;

    cuda_res = cusparseCscGet((cusparseSpMatDescr_t)spMatDescr, rows,
                                   cols, nnz, cscColOffsets, cscRowInd, cscValues,
                                   &cudaIdxRowType, &cudaIdxColType, &cudaIdxBase, &cudaType);
    *cscColOffsetsType = cusparseIndexTypeToUPTKsparseIndexType(cudaIdxRowType);
    *cscRowIndType = cusparseIndexTypeToUPTKsparseIndexType(cudaIdxColType);
    *idxBase = cusparseIndexBaseToUPTKsparseIndexBase(cudaIdxBase);
    *valueType = cudaDataTypeToUPTKDataType(cudaType);

    return cusparseStatusToUPTKsparseStatus(cuda_res);
}
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseConstCsrGet(UPTKsparseSpMatDescr_t spMatDescr,
                      int64_t *rows,
                      int64_t *cols,
                      int64_t *nnz,
                      const void **csrRowOffsets,
                      const void **csrColInd,
                      const void **csrValues,
                      UPTKsparseIndexType_t *csrRowOffsetsType,
                      UPTKsparseIndexType_t *csrColIndType,
                      UPTKsparseIndexBase_t *idxBase,
                      UPTKDataType *valueType)
{
    cusparseStatus_t cuda_res;
    cusparseIndexType_t cudaIdxRowType;
    cusparseIndexType_t cudaIdxColType;
    cusparseIndexBase_t cudaIdxBase;
    cudaDataType cudaType;

    cuda_res = cusparseConstCsrGet((cusparseSpMatDescr_t)spMatDescr, rows,
                                   cols, nnz, csrRowOffsets, csrColInd, csrValues,
                                   &cudaIdxRowType, &cudaIdxColType, &cudaIdxBase, &cudaType);
    *csrRowOffsetsType = cusparseIndexTypeToUPTKsparseIndexType(cudaIdxRowType);
    *csrColIndType = cusparseIndexTypeToUPTKsparseIndexType(cudaIdxColType);
    *idxBase = cusparseIndexBaseToUPTKsparseIndexBase(cudaIdxBase);
    *valueType = cudaDataTypeToUPTKDataType(cudaType);

    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseConstDnMatGet(UPTKsparseDnMatDescr_t dnMatDescr,
                        int64_t *rows,
                        int64_t *cols,
                        int64_t *ld,
                        const void **values,
                        UPTKDataType *type,
                        UPTKsparseOrder_t *order)
{
    cusparseStatus_t cuda_res;
    cudaDataType cudaType;
    cusparseOrder_t cudaOrder;
    cuda_res = cusparseConstDnMatGet((cusparseDnMatDescr_t)dnMatDescr,
                                     rows, cols, ld, values, &cudaType, &cudaOrder);
    *type = cudaDataTypeToUPTKDataType(cudaType);
    *order = cusparseOrderToUPTKsparseOrder(cudaOrder);

    return cusparseStatusToUPTKsparseStatus(cuda_res);
}
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseConstDnMatGetValues(UPTKsparseDnMatDescr_t dnMatDescr,
                              const void **values)
{
    return cusparseStatusToUPTKsparseStatus(cusparseConstDnMatGetValues((cusparseDnMatDescr_t)dnMatDescr, values));
}

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseConstDnVecGet(UPTKsparseDnVecDescr_t dnVecDescr,
                        int64_t *size,
                        const void **values,
                        UPTKDataType *valueType)
{
    cusparseStatus_t cuda_res;
    cudaDataType cudaType;
    cuda_res = cusparseConstDnVecGet((cusparseDnVecDescr_t)dnVecDescr, size, values, &cudaType);
    *valueType = cudaDataTypeToUPTKDataType(cudaType);

    return cusparseStatusToUPTKsparseStatus(cuda_res);
}
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseConstDnVecGetValues(UPTKsparseDnVecDescr_t dnVecDescr,
                              const void **values)
{
    return cusparseStatusToUPTKsparseStatus(cusparseConstDnVecGetValues((cusparseDnVecDescr_t)dnVecDescr, values));
}
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseConstSpMatGetValues(UPTKsparseSpMatDescr_t spMatDescr,
                              const void **values)
{
    return cusparseStatusToUPTKsparseStatus(cusparseConstSpMatGetValues((cusparseSpMatDescr_t)spMatDescr, values));
}
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseConstSpVecGet(UPTKsparseSpVecDescr_t spVecDescr,
                        int64_t *size,
                        int64_t *nnz,
                        const void **indices,
                        const void **values,
                        UPTKsparseIndexType_t *idxType,
                        UPTKsparseIndexBase_t *idxBase,
                        UPTKDataType *valueType)
{
    cusparseStatus_t cuda_res;
    cusparseIndexType_t cudaIdxType;
    cusparseIndexBase_t cudaIdxBase;
    cudaDataType cudaType;
    cuda_res = cusparseConstSpVecGet((cusparseConstSpVecDescr_t)spVecDescr,
                                     size, nnz, indices, values,
                                     &cudaIdxType, &cudaIdxBase, &cudaType);
    *idxType = cusparseIndexTypeToUPTKsparseIndexType(cudaIdxType);
    *idxBase = cusparseIndexBaseToUPTKsparseIndexBase(cudaIdxBase);
    *valueType = cudaDataTypeToUPTKDataType(cudaType);

    return cusparseStatusToUPTKsparseStatus(cuda_res);
}
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseConstSpVecGetValues(UPTKsparseSpVecDescr_t spVecDescr,
                              const void **values)
{
    return cusparseStatusToUPTKsparseStatus(cusparseConstSpVecGetValues((cusparseConstSpVecDescr_t)spVecDescr, values));
}
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateBsr(UPTKsparseSpMatDescr_t *spMatDescr,
                    int64_t brows,
                    int64_t bcols,
                    int64_t bnnz,
                    int64_t rowBlockSize,
                    int64_t colBlockSize,
                    void *bsrRowOffsets,
                    void *bsrColInd,
                    void *bsrValues,
                    UPTKsparseIndexType_t bsrRowOffsetsType,
                    UPTKsparseIndexType_t bsrColIndType,
                    UPTKsparseIndexBase_t idxBase,
                    UPTKDataType valueType,
                    UPTKsparseOrder_t order)
{
    cusparseStatus_t cuda_res;
    cusparseIndexType_t cudaIdxRowType = UPTKsparseIndexTypeTocusparseIndexType(bsrRowOffsetsType);
    cusparseIndexType_t cudaIdxColType = UPTKsparseIndexTypeTocusparseIndexType(bsrColIndType);
    ;
    cusparseIndexBase_t cudaIdxBase = UPTKsparseIndexBaseTocusparseIndexBase(idxBase);
    cudaDataType cudaType = UPTKDataTypeTocudaDataType(valueType);
    cusparseOrder_t cudaOrder = UPTKsparseOrderTocusparseOrder(order);
    cuda_res = cusparseCreateBsr((cusparseSpMatDescr_t *)spMatDescr, brows, bcols, bnnz,
                                 rowBlockSize, colBlockSize, bsrRowOffsets, bsrColInd, bsrValues,
                                 cudaIdxRowType, cudaIdxColType, cudaIdxBase, cudaType, cudaOrder);

    return cusparseStatusToUPTKsparseStatus(cuda_res);
}
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateConstBlockedEll(UPTKsparseSpMatDescr_t *spMatDescr,
                                int64_t rows,
                                int64_t cols,
                                int64_t ellBlockSize,
                                int64_t ellCols,
                                const void *ellColInd,
                                const void *ellValue,
                                UPTKsparseIndexType_t ellIdxType,
                                UPTKsparseIndexBase_t idxBase,
                                UPTKDataType valueType)
{
    cusparseStatus_t cuda_res;
    cusparseIndexType_t cudaIdxType = UPTKsparseIndexTypeTocusparseIndexType(ellIdxType);
    cusparseIndexBase_t cudaIdxBase = UPTKsparseIndexBaseTocusparseIndexBase(idxBase);
    cudaDataType cudaType = UPTKDataTypeTocudaDataType(valueType);
    cuda_res = cusparseCreateConstBlockedEll((cusparseSpMatDescr_t *)spMatDescr,
                                             rows, cols, ellBlockSize, ellCols, ellColInd,
                                             ellValue, cudaIdxType, cudaIdxBase, cudaType);

    return cusparseStatusToUPTKsparseStatus(cuda_res);
}
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateConstBsr(UPTKsparseSpMatDescr_t *spMatDescr,
                         int64_t brows,
                         int64_t bcols,
                         int64_t bnnz,
                         int64_t rowBlockDim,
                         int64_t colBlockDim,
                         const void *bsrRowOffsets,
                         const void *bsrColInd,
                         const void *bsrValues,
                         UPTKsparseIndexType_t bsrRowOffsetsType,
                         UPTKsparseIndexType_t bsrColIndType,
                         UPTKsparseIndexBase_t idxBase,
                         UPTKDataType valueType,
                         UPTKsparseOrder_t order)
{
    cusparseStatus_t cuda_res;
    cusparseIndexType_t cudaIdxRowType = UPTKsparseIndexTypeTocusparseIndexType(bsrRowOffsetsType);
    cusparseIndexType_t cudaIdxColType = UPTKsparseIndexTypeTocusparseIndexType(bsrColIndType);
    ;
    cusparseIndexBase_t cudaIdxBase = UPTKsparseIndexBaseTocusparseIndexBase(idxBase);
    cudaDataType cudaType = UPTKDataTypeTocudaDataType(valueType);
    cusparseOrder_t cudaOrder = UPTKsparseOrderTocusparseOrder(order);
    cuda_res = cusparseCreateConstBsr((cusparseSpMatDescr_t *)spMatDescr,
                                      brows, bcols, bnnz, rowBlockDim, colBlockDim,
                                      bsrRowOffsets, bsrColInd, bsrValues, cudaIdxRowType,
                                      cudaIdxColType, cudaIdxBase, cudaType, cudaOrder);

    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateConstCoo(UPTKsparseSpMatDescr_t *spMatDescr,
                         int64_t rows,
                         int64_t cols,
                         int64_t nnz,
                         const void *cooRowInd,
                         const void *cooColInd,
                         const void *cooValues,
                         UPTKsparseIndexType_t cooIdxType,
                         UPTKsparseIndexBase_t idxBase,
                         UPTKDataType valueType)
{
    cusparseStatus_t cuda_res;
    cusparseIndexType_t cudaIdxType = UPTKsparseIndexTypeTocusparseIndexType(cooIdxType);
    cusparseIndexBase_t cudaIdxBase = UPTKsparseIndexBaseTocusparseIndexBase(idxBase);
    cudaDataType cudaType = UPTKDataTypeTocudaDataType(valueType);
    cuda_res = cusparseCreateConstCoo((cusparseSpMatDescr_t *)spMatDescr,
                                      rows, cols, nnz, cooRowInd, cooColInd,
                                      cooValues, cudaIdxType, cudaIdxBase, cudaType);

    return cusparseStatusToUPTKsparseStatus(cuda_res);
}
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateConstCsc(UPTKsparseSpMatDescr_t *spMatDescr,
                         int64_t rows,
                         int64_t cols,
                         int64_t nnz,
                         const void *cscColOffsets,
                         const void *cscRowInd,
                         const void *cscValues,
                         UPTKsparseIndexType_t cscColOffsetsType,
                         UPTKsparseIndexType_t cscRowIndType,
                         UPTKsparseIndexBase_t idxBase,
                         UPTKDataType valueType)
{
    cusparseStatus_t cuda_res;
    cusparseIndexType_t cudaIdxRowType = UPTKsparseIndexTypeTocusparseIndexType(cscColOffsetsType);
    cusparseIndexType_t cudaIdxColType = UPTKsparseIndexTypeTocusparseIndexType(cscRowIndType);
    ;
    cusparseIndexBase_t cudaIdxBase = UPTKsparseIndexBaseTocusparseIndexBase(idxBase);
    cudaDataType cudaType = UPTKDataTypeTocudaDataType(valueType);
    cuda_res = cusparseCreateConstCsc((cusparseSpMatDescr_t *)spMatDescr, rows,
                                      cols, nnz, cscColOffsets, cscRowInd, cscValues,
                                      cudaIdxRowType, cudaIdxColType, cudaIdxBase, cudaType);

    return cusparseStatusToUPTKsparseStatus(cuda_res);
}*/

/*UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateConstCsr(UPTKsparseSpMatDescr_t *spMatDescr,
                         int64_t rows,
                         int64_t cols,
                         int64_t nnz,
                         const void *csrRowOffsets,
                         const void *csrColInd,
                         const void *csrValues,
                         UPTKsparseIndexType_t csrRowOffsetsType,
                         UPTKsparseIndexType_t csrColIndType,
                         UPTKsparseIndexBase_t idxBase,
                         UPTKDataType valueType)
{
    cusparseStatus_t cuda_res;
    cusparseIndexType_t cudaIdxRowType = UPTKsparseIndexTypeTocusparseIndexType(csrRowOffsetsType);
    cusparseIndexType_t cudaIdxColType = UPTKsparseIndexTypeTocusparseIndexType(csrColIndType);
    ;
    cusparseIndexBase_t cudaIdxBase = UPTKsparseIndexBaseTocusparseIndexBase(idxBase);
    cudaDataType cudaType = UPTKDataTypeTocudaDataType(valueType);
    cuda_res = cusparseCreateConstCsr((cusparseSpMatDescr_t *)spMatDescr, rows,
                                      cols, nnz, csrRowOffsets, csrColInd, csrValues,
                                      cudaIdxRowType, cudaIdxColType, cudaIdxBase, cudaType);

    return cusparseStatusToUPTKsparseStatus(cuda_res);
}
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateConstDnMat(UPTKsparseDnMatDescr_t *dnMatDescr,
                           int64_t rows,
                           int64_t cols,
                           int64_t ld,
                           const void *values,
                           UPTKDataType valueType,
                           UPTKsparseOrder_t order)
{
    cusparseStatus_t cuda_res;
    cudaDataType cudaType = UPTKDataTypeTocudaDataType(valueType);
    cusparseOrder_t cudaOrder = UPTKsparseOrderTocusparseOrder(order);
    cuda_res = cusparseCreateConstDnMat((cusparseDnMatDescr_t *)dnMatDescr,
                                        rows, cols, ld, values, cudaType, cudaOrder);

    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateConstDnVec(UPTKsparseDnVecDescr_t *dnVecDescr,
                           int64_t size,
                           const void *values,
                           UPTKDataType valueType)
{
    cusparseStatus_t cuda_res;
    cudaDataType cudaType = UPTKDataTypeTocudaDataType(valueType);
    cuda_res = cusparseCreateConstDnVec((cusparseDnVecDescr_t *)dnVecDescr, size, values, cudaType);

    return cusparseStatusToUPTKsparseStatus(cuda_res);
}
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateConstSlicedEll(UPTKsparseSpMatDescr_t *spMatDescr,
                               int64_t rows,
                               int64_t cols,
                               int64_t nnz,
                               int64_t sellValuesSize,
                               int64_t sliceSize,
                               const void *sellSliceOffsets,
                               const void *sellColInd,
                               const void *sellValues,
                               UPTKsparseIndexType_t sellSliceOffsetsType,
                               UPTKsparseIndexType_t sellColIndType,
                               UPTKsparseIndexBase_t idxBase,
                               UPTKDataType valueType)
{
    cusparseStatus_t cuda_res;
    cusparseIndexType_t cudaIdxRowType = UPTKsparseIndexTypeTocusparseIndexType(sellSliceOffsetsType);
    cusparseIndexType_t cudaIdxColType = UPTKsparseIndexTypeTocusparseIndexType(sellColIndType);
    ;
    cusparseIndexBase_t cudaIdxBase = UPTKsparseIndexBaseTocusparseIndexBase(idxBase);
    cudaDataType cudaType = UPTKDataTypeTocudaDataType(valueType);

    cuda_res = cusparseCreateConstSlicedEll((cusparseSpMatDescr_t *)spMatDescr,
                                            rows, cols, nnz, sellValuesSize, sliceSize,
                                            sellSliceOffsets, sellColInd, sellValues,
                                            cudaIdxRowType, cudaIdxColType, cudaIdxBase, cudaType);

    return cusparseStatusToUPTKsparseStatus(cuda_res);
}
UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateConstSpVec(UPTKsparseSpVecDescr_t *spVecDescr,
                           int64_t size,
                           int64_t nnz,
                           const void *indices,
                           const void *values,
                           UPTKsparseIndexType_t idxType,
                           UPTKsparseIndexBase_t idxBase,
                           UPTKDataType valueType)
{
    cusparseStatus_t cuda_res;
    cusparseIndexType_t cudaIdxType = UPTKsparseIndexTypeTocusparseIndexType(idxType);
    ;
    cusparseIndexBase_t cudaIdxBase = UPTKsparseIndexBaseTocusparseIndexBase(idxBase);
    cudaDataType cudaType = UPTKDataTypeTocudaDataType(valueType);

    cuda_res = cusparseCreateConstSpVec((cusparseConstSpVecDescr_t *)spVecDescr, size, nnz,
                                        indices, values, cudaIdxType, cudaIdxBase, cudaType);

    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseCreateSlicedEll(UPTKsparseSpMatDescr_t *spMatDescr,
                          int64_t rows,
                          int64_t cols,
                          int64_t nnz,
                          int64_t sellValuesSize,
                          int64_t sliceSize,
                          void *sellSliceOffsets,
                          void *sellColInd,
                          void *sellValues,
                          UPTKsparseIndexType_t sellSliceOffsetsType,
                          UPTKsparseIndexType_t sellColIndType,
                          UPTKsparseIndexBase_t idxBase,
                          UPTKDataType valueType)
{
    cusparseStatus_t cuda_res;
    cusparseIndexType_t cudaIdxRowType = UPTKsparseIndexTypeTocusparseIndexType(sellSliceOffsetsType);
    cusparseIndexType_t cudaIdxColType = UPTKsparseIndexTypeTocusparseIndexType(sellColIndType);
    ;
    cusparseIndexBase_t cudaIdxBase = UPTKsparseIndexBaseTocusparseIndexBase(idxBase);
    cudaDataType cudaType = UPTKDataTypeTocudaDataType(valueType);

    cuda_res = cusparseCreateSlicedEll((cusparseSpMatDescr_t *)spMatDescr,
                                       rows, cols, nnz, sellValuesSize, sliceSize,
                                       sellSliceOffsets, sellColInd, sellValues,
                                       cudaIdxRowType, cudaIdxColType, cudaIdxBase, cudaType);

    return cusparseStatusToUPTKsparseStatus(cuda_res);
}

UPTKsparseStatus_t UPTKSPARSEAPI
UPTKsparseSpMV_preprocess(UPTKsparseHandle_t handle,
                          UPTKsparseOperation_t opA,
                          const void *alpha,
                          UPTKsparseSpMatDescr_t matA,
                          UPTKsparseDnVecDescr_t vecX,
                          const void *beta,
                          UPTKsparseDnVecDescr_t vecY,
                          UPTKDataType computeType,
                          UPTKsparseSpMVAlg_t alg,
                          void *externalBuffer)
{
    cusparseStatus_t cuda_res;
    cusparseOperation_t cudaOperation = UPTKsparseOperationTocusparseOperation(opA);
    cudaDataType cudaType = UPTKDataTypeTocudaDataType(computeType);
    cusparseSpMVAlg_t cudaSpAlg = UPTKsparseSpMVAlgTocusparseSpMVAlg(alg);
    cuda_res = cusparseSpMV_preprocess((cusparseHandle_t)handle,
                                       cudaOperation, alpha, (cusparseSpMatDescr_t)matA, (cusparseDnVecDescr_t)vecX, beta,
                                       (cusparseDnVecDescr_t)vecY, cudaType, cudaSpAlg, externalBuffer);

    return cusparseStatusToUPTKsparseStatus(cuda_res);
}*/
