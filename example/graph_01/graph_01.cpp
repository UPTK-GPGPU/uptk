
#include <stdio.h>
#include <string.h>  // 用于 memset
#include <hip/hip_runtime.h>
#include <UPTK_runtime_api.h>

// 向量加法内核
__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int N = 256;
    size_t size = N * sizeof(float);
    UPTKError_t err;
    
    // 分配主机内存
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    
    // 初始化数据
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    
    // 分配设备内存
    float *d_a, *d_b, *d_c;
    UPTKMalloc(&d_a, size);
    UPTKMalloc(&d_b, size);
    UPTKMalloc(&d_c, size);
    
    // 1. 创建图 - UPTKGraphCreate
    UPTKGraph_t graph;
    err = UPTKGraphCreate(&graph, 0);
    if (err != UPTKSuccess) {
        printf("UPTKGraphCreate failed: %s\n", UPTKGetErrorString(err));
        free(h_a);
        free(h_b);
        free(h_c);
        return -1;
    }
    printf("Graph created successfully\n");
    
    // 创建图节点
    UPTKGraphNode_t memcpyH2D_node1, memcpyH2D_node2;
    UPTKGraphNode_t kernel_node;
    UPTKGraphNode_t memcpyD2H_node;
    
    // 2. 使用 UPTKGraphAddNode 添加内存拷贝节点 (主机到设备)
    UPTKGraphNodeParams* nodeParams = (UPTKGraphNodeParams*)malloc(sizeof(UPTKGraphNodeParams));
    if (!nodeParams) {
        printf("Failed to allocate nodeParams\n");
        UPTKGraphDestroy(graph);
        free(h_a);
        free(h_b);
        free(h_c);
        return -1;
    }
    memset(nodeParams, 0, sizeof(UPTKGraphNodeParams));
    nodeParams->type = UPTKGraphNodeTypeMemcpy;
    
    // 设置内存拷贝参数
    UPTKMemcpyNodeParams memcpyParams1;
    memset(&memcpyParams1, 0, sizeof(memcpyParams1));
    memset(&memcpyParams1.copyParams, 0, sizeof(memcpyParams1.copyParams));
    
    memcpyParams1.copyParams.srcArray = NULL;
    memcpyParams1.copyParams.srcPos = make_UPTKPos(0, 0, 0);
    memcpyParams1.copyParams.srcPtr = make_UPTKPitchedPtr(h_a, size, N, 1);
    memcpyParams1.copyParams.dstArray = NULL;
    memcpyParams1.copyParams.dstPos = make_UPTKPos(0, 0, 0);
    memcpyParams1.copyParams.dstPtr = make_UPTKPitchedPtr(d_a, size, N, 1);
    memcpyParams1.copyParams.extent = make_UPTKExtent(size, 1, 1);
    memcpyParams1.copyParams.kind = UPTKMemcpyHostToDevice;
    
    nodeParams->memcpy = memcpyParams1;
    err = UPTKGraphAddNode(&memcpyH2D_node1, graph, NULL, 0, nodeParams);
    if (err != UPTKSuccess) {
        printf("UPTKGraphAddNode for memcpy failed: %s\n", UPTKGetErrorString(err));
        free(nodeParams);
        UPTKGraphDestroy(graph);
        free(h_a);
        free(h_b);
        free(h_c);
        UPTKFree(d_a);
        UPTKFree(d_b);
        UPTKFree(d_c);
        return -1;
    }
    printf("Memory copy node (H2D) added using UPTKGraphAddNode\n");
    
    // 添加第二个内存拷贝节点
    memset(nodeParams, 0, sizeof(UPTKGraphNodeParams));
    nodeParams->type = UPTKGraphNodeTypeMemcpy;
    
    UPTKMemcpyNodeParams memcpyParams2;
    memset(&memcpyParams2, 0, sizeof(memcpyParams2));
    memset(&memcpyParams2.copyParams, 0, sizeof(memcpyParams2.copyParams));
    
    memcpyParams2.copyParams.srcArray = NULL;
    memcpyParams2.copyParams.srcPos = make_UPTKPos(0, 0, 0);
    memcpyParams2.copyParams.srcPtr = make_UPTKPitchedPtr(h_b, size, N, 1);
    memcpyParams2.copyParams.dstArray = NULL;
    memcpyParams2.copyParams.dstPos = make_UPTKPos(0, 0, 0);
    memcpyParams2.copyParams.dstPtr = make_UPTKPitchedPtr(d_b, size, N, 1);
    memcpyParams2.copyParams.extent = make_UPTKExtent(size, 1, 1);
    memcpyParams2.copyParams.kind = UPTKMemcpyHostToDevice;
    
    nodeParams->memcpy = memcpyParams2;
    err = UPTKGraphAddNode(&memcpyH2D_node2, graph, NULL, 0, nodeParams);
    if (err != UPTKSuccess) {
        printf("UPTKGraphAddNode for memcpy failed: %s\n", UPTKGetErrorString(err));
        free(nodeParams);
        UPTKGraphDestroy(graph);
        free(h_a);
        free(h_b);
        free(h_c);
        UPTKFree(d_a);
        UPTKFree(d_b);
        UPTKFree(d_c);
        return -1;
    }
    printf("Second memory copy node (H2D) added using UPTKGraphAddNode\n");
    
    // 3. 使用 UPTKGraphAddNode 添加内核节点
    memset(nodeParams, 0, sizeof(UPTKGraphNodeParams));
    nodeParams->type = UPTKGraphNodeTypeKernel;
    
    UPTKKernelNodeParamsV2 kernelParams;
    memset(&kernelParams, 0, sizeof(kernelParams));
    
    void* kernelArgs[] = {(void*)&d_a, (void*)&d_b, (void*)&d_c, (void*)&N};
    kernelParams.func = (void*)vectorAdd;
    kernelParams.gridDim = dim3((N + 255) / 256, 1, 1);
    kernelParams.blockDim = dim3(256, 1, 1);
    kernelParams.sharedMemBytes = 0;
    kernelParams.kernelParams = (void**)kernelArgs;
    kernelParams.extra = NULL;
    
    nodeParams->kernel = kernelParams;
    err = UPTKGraphAddNode(&kernel_node, graph, NULL, 0, nodeParams);
    if (err != UPTKSuccess) {
        printf("UPTKGraphAddNode for kernel failed: %s\n", UPTKGetErrorString(err));
        free(nodeParams);
        UPTKGraphDestroy(graph);
        free(h_a);
        free(h_b);
        free(h_c);
        UPTKFree(d_a);
        UPTKFree(d_b);
        UPTKFree(d_c);
        return -1;
    }
    printf("Kernel node added using UPTKGraphAddNode\n");
    
    // 添加内存拷贝节点 (设备到主机)
    memset(nodeParams, 0, sizeof(UPTKGraphNodeParams));
    nodeParams->type = UPTKGraphNodeTypeMemcpy;
    
    UPTKMemcpyNodeParams memcpyParams3;
    memset(&memcpyParams3, 0, sizeof(memcpyParams3));
    memset(&memcpyParams3.copyParams, 0, sizeof(memcpyParams3.copyParams));
    
    memcpyParams3.copyParams.srcArray = NULL;
    memcpyParams3.copyParams.srcPos = make_UPTKPos(0, 0, 0);
    memcpyParams3.copyParams.srcPtr = make_UPTKPitchedPtr(d_c, size, N, 1);
    memcpyParams3.copyParams.dstArray = NULL;
    memcpyParams3.copyParams.dstPos = make_UPTKPos(0, 0, 0);
    memcpyParams3.copyParams.dstPtr = make_UPTKPitchedPtr(h_c, size, N, 1);
    memcpyParams3.copyParams.extent = make_UPTKExtent(size, 1, 1);
    memcpyParams3.copyParams.kind = UPTKMemcpyDeviceToHost;
    
    nodeParams->memcpy = memcpyParams3;
    
    err = UPTKGraphAddNode(&memcpyD2H_node, graph, NULL, 0, nodeParams);
    if (err != UPTKSuccess) {
        printf("UPTKGraphAddNode for memcpy failed: %s\n", UPTKGetErrorString(err));
        free(nodeParams);
        UPTKGraphDestroy(graph);
        free(h_a);
        free(h_b);
        free(h_c);
        UPTKFree(d_a);
        UPTKFree(d_b);
        UPTKFree(d_c);
        return -1;
    }
    printf("Memory copy node (D2H) added using UPTKGraphAddNode\n");
    
    // 4. 添加依赖关系 - UPTKGraphAddDependencies
    UPTKGraphNode_t fromNodes1[] = {memcpyH2D_node1, memcpyH2D_node2};
    UPTKGraphNode_t toNodes1[] = {kernel_node, kernel_node}; // 每个源节点对应一个目标节点
    
    // 设置依赖: 内存拷贝 -> 内核
    err = UPTKGraphAddDependencies(graph, fromNodes1, toNodes1, 2);
    if (err != UPTKSuccess) {
        printf("UPTKGraphAddDependencies failed: %s\n", UPTKGetErrorString(err));
        free(nodeParams);
        UPTKGraphDestroy(graph);
        free(h_a);
        free(h_b);
        free(h_c);
        UPTKFree(d_a);
        UPTKFree(d_b);
        UPTKFree(d_c);
        return -1;
    }
    printf("Dependencies added: memcpy -> kernel\n");
    
    // 设置依赖: 内核 -> 内存拷贝回传
    UPTKGraphNode_t fromNodes2[] = {kernel_node};
    UPTKGraphNode_t toNodes2[] = {memcpyD2H_node};
    
    err = UPTKGraphAddDependencies(graph, fromNodes2, toNodes2, 1);
    if (err != UPTKSuccess) {
        printf("UPTKGraphAddDependencies failed: %s\n", UPTKGetErrorString(err));
        free(nodeParams);
        UPTKGraphDestroy(graph);
        free(h_a);
        free(h_b);
        free(h_c);
        UPTKFree(d_a);
        UPTKFree(d_b);
        UPTKFree(d_c);
        return -1;
    }
    printf("Dependencies added: kernel -> memcpy (D2H)\n");
    
    // 5. 获取图中节点 - UPTKGraphGetNodes
    UPTKGraphNode_t* nodes = NULL;
    size_t numNodes = 0;
    err = UPTKGraphGetNodes(graph, nodes, &numNodes);
    if (err != UPTKSuccess) {
        printf("UPTKGraphGetNodes failed: %s\n", UPTKGetErrorString(err));
        free(nodeParams);
        UPTKGraphDestroy(graph);
        free(h_a);
        free(h_b);
        free(h_c);
        UPTKFree(d_a);
        UPTKFree(d_b);
        UPTKFree(d_c);
        return -1;
    }
    
    nodes = (UPTKGraphNode_t*)malloc(numNodes * sizeof(UPTKGraphNode_t));
    err = UPTKGraphGetNodes(graph, nodes, &numNodes);
    if (err != UPTKSuccess) {
        printf("UPTKGraphGetNodes failed: %s\n", UPTKGetErrorString(err));
        free(nodeParams);
        UPTKGraphDestroy(graph);
        free(h_a);
        free(h_b);
        free(h_c);
        free(nodes);
        UPTKFree(d_a);
        UPTKFree(d_b);
        UPTKFree(d_c);
        return -1;
    }
    
    printf("Graph has %zu nodes:\n", numNodes);
    
    // 6. 获取节点类型 - UPTKGraphNodeGetType
    for (size_t i = 0; i < numNodes; i++) {
        UPTKGraphNodeType nodeType;
        err = UPTKGraphNodeGetType(nodes[i], &nodeType);
        if (err != UPTKSuccess) {
            printf("UPTKGraphNodeGetType failed: %s\n", UPTKGetErrorString(err));
            free(nodeParams);
            UPTKGraphDestroy(graph);
            free(h_a);
            free(h_b);
            free(h_c);
            free(nodes);
            UPTKFree(d_a);
            UPTKFree(d_b);
            UPTKFree(d_c);
            return -1;
        }
        
        const char* typeName = "Unknown";
        switch (nodeType) {
            case UPTKGraphNodeTypeKernel: typeName = "Kernel"; break;
            case UPTKGraphNodeTypeMemcpy: typeName = "Memcpy"; break;
            case UPTKGraphNodeTypeMemset: typeName = "Memset"; break;
            default: break;
        }
        printf("  Node %zu: %s\n", i, typeName);
    }
    
    // 7. 移除依赖关系 - UPTKGraphRemoveDependencies
    err = UPTKGraphRemoveDependencies(graph, fromNodes1, toNodes1, 2);
    if (err != UPTKSuccess) {
        printf("UPTKGraphRemoveDependencies failed: %s\n", UPTKGetErrorString(err));
        free(nodeParams);
        UPTKGraphDestroy(graph);
        free(h_a);
        free(h_b);
        free(h_c);
        free(nodes);
        UPTKFree(d_a);
        UPTKFree(d_b);
        UPTKFree(d_c);
        return -1;
    }
    printf("Dependencies removed: memcpy -> kernel\n");
    
    // 8. 重新添加依赖 - UPTKGraphAddDependencies
    err = UPTKGraphAddDependencies(graph, fromNodes1, toNodes1, 2);
    if (err != UPTKSuccess) {
        printf("UPTKGraphAddDependencies failed: %s\n", UPTKGetErrorString(err));
        free(nodeParams);
        UPTKGraphDestroy(graph);
        free(h_a);
        free(h_b);
        free(h_c);
        free(nodes);
        UPTKFree(d_a);
        UPTKFree(d_b);
        UPTKFree(d_c);
        return -1;
    }
    printf("Dependencies re-added: memcpy -> kernel\n");
    
    // 9. 克隆图 - UPTKGraphClone
    UPTKGraph_t clonedGraph;
    err = UPTKGraphClone(&clonedGraph, graph);
    if (err != UPTKSuccess) {
        printf("UPTKGraphClone failed: %s\n", UPTKGetErrorString(err));
        free(nodeParams);
        UPTKGraphDestroy(graph);
        free(h_a);
        free(h_b);
        free(h_c);
        free(nodes);
        UPTKFree(d_a);
        UPTKFree(d_b);
        UPTKFree(d_c);
        return -1;
    }
    printf("Graph cloned successfully\n");
    
    // 10. 在克隆图中移除依赖并销毁节点
    UPTKGraphNode_t* clonedNodes = NULL;
    size_t clonedNumNodes = 0;
    err = UPTKGraphGetNodes(clonedGraph, clonedNodes, &clonedNumNodes);
    clonedNodes = (UPTKGraphNode_t*)malloc(clonedNumNodes * sizeof(UPTKGraphNode_t));
    err = UPTKGraphGetNodes(clonedGraph, clonedNodes, &clonedNumNodes);
    
    UPTKGraphNode_t clonedFromNodes[] = {clonedNodes[0], clonedNodes[1]};
    UPTKGraphNode_t clonedToNodes[] = {clonedNodes[2], clonedNodes[2]}; // 两个源节点指向同一个目标节点
    
    err = UPTKGraphRemoveDependencies(clonedGraph, clonedFromNodes, clonedToNodes, 2);
    if (err != UPTKSuccess) {
        printf("UPTKGraphRemoveDependencies failed: %s\n", UPTKGetErrorString(err));
        free(nodeParams);
        UPTKGraphDestroy(graph);
        UPTKGraphDestroy(clonedGraph);
        free(h_a);
        free(h_b);
        free(h_c);
        free(nodes);
        free(clonedNodes);
        UPTKFree(d_a);
        UPTKFree(d_b);
        UPTKFree(d_c);
        return -1;
    }
    printf("Dependencies removed in cloned graph\n");
    
    // 销毁克隆图中的第一个节点
    err = UPTKGraphDestroyNode(clonedNodes[0]);
    if (err != UPTKSuccess) {
        printf("UPTKGraphDestroyNode failed: %s\n", UPTKGetErrorString(err));
        free(nodeParams);
        UPTKGraphDestroy(graph);
        UPTKGraphDestroy(clonedGraph);
        free(h_a);
        free(h_b);
        free(h_c);
        free(nodes);
        free(clonedNodes);
        UPTKFree(d_a);
        UPTKFree(d_b);
        UPTKFree(d_c);
        return -1;
    }
    printf("Node destroyed in cloned graph\n");
    
    // 实例化并执行原始图
    UPTKGraphExec_t graphExec;
    err = UPTKGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    if (err != UPTKSuccess) {
        printf("UPTKGraphInstantiate failed: %s\n", UPTKGetErrorString(err));
        free(nodeParams);
        UPTKGraphDestroy(graph);
        UPTKGraphDestroy(clonedGraph);
        free(h_a);
        free(h_b);
        free(h_c);
        free(nodes);
        free(clonedNodes);
        UPTKFree(d_a);
        UPTKFree(d_b);
        UPTKFree(d_c);
        return -1;
    }
    
    // 第一次执行图
    err = UPTKGraphLaunch(graphExec, 0);
    if (err != UPTKSuccess) {
        printf("First UPTKGraphLaunch failed: %s\n", UPTKGetErrorString(err));
        free(nodeParams);
        UPTKGraphDestroy(graph);
        UPTKGraphDestroy(clonedGraph);
        free(h_a);
        free(h_b);
        free(h_c);
        free(nodes);
        free(clonedNodes);
        UPTKFree(d_a);
        UPTKFree(d_b);
        UPTKFree(d_c);
        return -1;
    }
    
    UPTKDeviceSynchronize();
    printf("First graph execution completed successfully\n");
    
    // 验证第一次执行结果
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_c[i] - (h_a[i] + h_b[i])) > 1e-5) {
            success = false;
            break;
        }
    }
    
    if (success) {
        printf("First execution results verified successfully!\n");
    } else {
        printf("First execution results verification failed!\n");
    }
    
    // 12. 使用 UPTKGraphExecUpdate 更新图执行实例
    printf("Updating graph execution instance...\n");
    UPTKGraphNode_t errorNode;
    UPTKGraphExecUpdateResult updateResult;

    err = UPTKGraphExecUpdate(graphExec, graph, &errorNode, &updateResult);
    if (err != UPTKSuccess) {
        printf("UPTKGraphExecUpdate failed: %s\n", UPTKGetErrorString(err));
    } else {
        printf("UPTKGraphExecUpdate completed with result: %d\n", updateResult);
        
        // 如果更新成功，再次执行图
        if (updateResult == UPTKGraphExecUpdateSuccess) {
            printf("Executing updated graph...\n");
            err = UPTKGraphLaunch(graphExec, 0);
            if (err != UPTKSuccess) {
                printf("Second UPTKGraphLaunch failed: %s\n", UPTKGetErrorString(err));
            } else {
                UPTKDeviceSynchronize();
                printf("Second graph execution completed successfully\n");
                
                // 验证第二次执行结果
                bool secondSuccess = true;
                for (int i = 0; i < N; i++) {
                    if (fabs(h_c[i] - (h_a[i] + h_b[i])) > 1e-5) {
                        secondSuccess = false;
                        break;
                    }
                }
                
                if (secondSuccess) {
                    printf("Second execution results verified successfully!\n");
                } else {
                    printf("Second execution results verification failed!\n");
                }
            }
        } else if (updateResult == UPTKGraphExecUpdateError) {
            printf("Graph update failed. Error node type: ");
            UPTKGraphNodeType errorNodeType;
            err = UPTKGraphNodeGetType(errorNode, &errorNodeType);
            if (err == UPTKSuccess) {
                const char* typeName = "Unknown";
                switch (errorNodeType) {
                    case UPTKGraphNodeTypeKernel: typeName = "Kernel"; break;
                    case UPTKGraphNodeTypeMemcpy: typeName = "Memcpy"; break;
                    case UPTKGraphNodeTypeMemset: typeName = "Memset"; break;
                    default: break;
                }
                printf("%s\n", typeName);
            } else {
                printf("Failed to get error node type\n");
            }
        }
    }
    
    // 13. 使用 UPTKGraphExecDestroy 销毁图执行实例
    printf("Destroying graph execution instance...\n");
    err = UPTKGraphExecDestroy(graphExec);
    if (err != UPTKSuccess) {
        printf("UPTKGraphExecDestroy failed: %s\n", UPTKGetErrorString(err));
    } else {
        printf("Graph execution instance destroyed successfully\n");
    }
    
    // 11. 销毁图 - UPTKGraphDestroy
    err = UPTKGraphDestroy(graph);
    if (err != UPTKSuccess) {
        printf("UPTKGraphDestroy failed: %s\n", UPTKGetErrorString(err));
        free(nodeParams);
        UPTKGraphDestroy(clonedGraph);
        free(h_a);
        free(h_b);
        free(h_c);
        free(nodes);
        free(clonedNodes);
        UPTKFree(d_a);
        UPTKFree(d_b);
        UPTKFree(d_c);
        return -1;
    }
    printf("Original graph destroyed\n");
    
    err = UPTKGraphDestroy(clonedGraph);
    if (err != UPTKSuccess) {
        printf("UPTKGraphDestroy failed: %s\n", UPTKGetErrorString(err));
        free(nodeParams);
        free(h_a);
        free(h_b);
        free(h_c);
        free(nodes);
        free(clonedNodes);
        UPTKFree(d_a);
        UPTKFree(d_b);
        UPTKFree(d_c);
        return -1;
    }
    printf("Cloned graph destroyed\n");
    
    // 清理资源
    free(h_a);
    free(h_b);
    free(h_c);
    free(nodes);
    free(clonedNodes);
    free(nodeParams);
    UPTKFree(d_a);
    UPTKFree(d_b);
    UPTKFree(d_c);
    
    printf("All UPTK Graph operations completed successfully!\n");
    return 0;
}
