#!/usr/bin/env node
/** Node-based tests for the CUDA -> UPTK converter. */

const fs = require("fs");
const path = require("path");
const { loadApiMap, convertSource, normalize } = require("./convert_lib");

const ROOT = path.resolve(__dirname, "..", "..");
const CUDA_DEMO = path.join(
  ROOT,
  "example_port",
  "cuda_port_demo",
  "MyDemo_CUDA",
  "MyDemo.cu"
);
const UPTK_DEMO = path.join(
  ROOT,
  "example_port",
  "cuda_port_demo",
  "MyDemo_UPTK",
  "MyDemo.cu"
);

function main() {
  const apiMap = loadApiMap();
  let passed = 0;
  let failed = 0;

  function ok(name) {
    console.log(`PASS: ${name}`);
    passed += 1;
  }
  function fail(name, detail) {
    console.error(`FAIL: ${name}`);
    if (detail) console.error(detail);
    failed += 1;
  }

  const source = fs.readFileSync(CUDA_DEMO, "utf8");
  const expected = normalize(fs.readFileSync(UPTK_DEMO, "utf8"));
  const converted = normalize(convertSource(source, apiMap));

  if (converted === expected) ok("demo matches reference");
  else fail("demo matches reference", "output differs from MyDemo_UPTK");

  const kernel = `
__global__ void add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}
`;
  if (convertSource(kernel, apiMap) === kernel) ok("kernel code unchanged");
  else fail("kernel code unchanged");

  const sample = convertSource(
    "#include <cuda_runtime_api.h>\nint main(){ cudaMalloc(nullptr,0); return 0; }\n",
    apiMap
  );
  if (sample.includes("UPTKMalloc") && sample.includes("#include <UPTK_runtime_api.h>")) {
    ok("runtime sample conversion");
  } else {
    fail("runtime sample conversion");
  }

  console.log(`\nResults: ${passed} passed, ${failed} failed`);
  process.exit(failed ? 1 : 0);
}

main();
