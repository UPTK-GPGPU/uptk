#!/usr/bin/env node
/** Generate CUDA->UPTK API mappings from uptk source convert files. */

const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..", "..");
const RUNTIME_CONVERT = path.join(ROOT, "src", "runtime", "runtime_fun_convert.cpp");
const DRIVER_CONVERT = path.join(ROOT, "src", "driver", "driver_fun_convert.cpp");
const OUTPUT = path.join(__dirname, "api_map.json");

const RUNTIME_DEF = /^\s*(?:__host__\s+)?UPTKError(?:_t)?\s+(UPTK[A-Za-z0-9_]+)\s*\(/gm;
const RUNTIME_CALL = /\bcuda([A-Za-z0-9_]+)\s*\(/g;
const DRIVER_DEF = /^\s*UPTKError\s+(UP[A-Za-z0-9_]+)\s*\(/gm;
const DRIVER_CALL = /\bcu([A-Za-z0-9_]+)\s*\(/g;

const DRIVER_TYPE_RULES = {
  CUresult: "UPTKError",
  CUDA_SUCCESS: "UPTKSuccess",
  CUdeviceptr: "UPTKdeviceptr",
  CUdeviceptr_v2: "UPTKdeviceptr",
};

function collectPairs(sourcePath, defPattern, callPattern, uptkPrefix, cudaPrefix) {
  const text = fs.readFileSync(sourcePath, "utf8");
  const mapping = {};
  const fallbacks = {};

  for (const match of text.matchAll(defPattern)) {
    const uptkName = match[1];
    const blockEnd = text.indexOf("\n}", match.index);
    if (blockEnd === -1) continue;
    const block = text.slice(match.index, blockEnd);
    const calls = [...block.matchAll(callPattern)].map((m) => m[1]);
    if (!calls.length) continue;
    const cudaName = `${cudaPrefix}${calls[0]}`;
    const canonical = `${uptkPrefix}${cudaName.slice(cudaPrefix.length)}`;
    if (uptkName === canonical) mapping[cudaName] = uptkName;
    else if (!(cudaName in fallbacks)) fallbacks[cudaName] = uptkName;
  }

  for (const [cudaName, uptkName] of Object.entries(fallbacks)) {
    if (!(cudaName in mapping)) mapping[cudaName] = uptkName;
  }

  return mapping;
}

const HEADER_NAME_SUFFIXES = ["_runtime_api", "_runtime", "_driver_types", "_library_types"];

function isHeaderName(cudaToken) {
  return HEADER_NAME_SUFFIXES.some((suffix) => cudaToken.endsWith(suffix));
}

function collectRuntimeTypes(runtimeFunctions) {
  const header = fs.readFileSync(path.join(ROOT, "include", "UPTK_runtime_api.h"), "utf8");
  const mapping = {};
  for (const token of header.match(/\bUPTK[A-Za-z0-9_]+\b/g) || []) {
    const cudaToken = `cuda${token.slice(4)}`;
    if (isHeaderName(cudaToken)) continue;
    if (cudaToken in runtimeFunctions) continue;
    mapping[cudaToken] = token;
  }
  return mapping;
}

function collectDriverTypes() {
  const header = fs.readFileSync(path.join(ROOT, "include", "UPTK.h"), "utf8");
  const mapping = { ...DRIVER_TYPE_RULES };
  for (const token of header.match(/\bUPTK[A-Za-z0-9_]+\b/g) || []) {
    mapping[`CU${token.slice(4)}`] = token;
  }
  return mapping;
}

function sortByKeyLengthDesc(obj) {
  return Object.fromEntries(
    Object.entries(obj).sort((a, b) => b[0].length - a[0].length)
  );
}

function generate() {
  const runtimeFunctions = collectPairs(
    RUNTIME_CONVERT,
    RUNTIME_DEF,
    RUNTIME_CALL,
    "UPTK",
    "cuda"
  );
  const driverFunctions = collectPairs(
    DRIVER_CONVERT,
    DRIVER_DEF,
    DRIVER_CALL,
    "UP",
    "cu"
  );

  for (const cudaName of Object.keys(runtimeFunctions)) {
    runtimeFunctions[cudaName] ||= `UPTK${cudaName.slice(4)}`;
  }
  for (const cuName of Object.keys(driverFunctions)) {
    driverFunctions[cuName] ||= `UP${cuName.slice(2)}`;
  }

  return {
    runtime_functions: sortByKeyLengthDesc(runtimeFunctions),
    driver_functions: sortByKeyLengthDesc(driverFunctions),
    runtime_types: sortByKeyLengthDesc(collectRuntimeTypes(runtimeFunctions)),
    driver_types: sortByKeyLengthDesc(collectDriverTypes()),
  };
}

const data = generate();
fs.writeFileSync(OUTPUT, `${JSON.stringify(data, null, 2)}\n`, "utf8");
console.log(`Wrote ${OUTPUT}`);
console.log(
  `runtime_functions=${Object.keys(data.runtime_functions).length}, ` +
    `driver_functions=${Object.keys(data.driver_functions).length}, ` +
    `runtime_types=${Object.keys(data.runtime_types).length}, ` +
    `driver_types=${Object.keys(data.driver_types).length}`
);
