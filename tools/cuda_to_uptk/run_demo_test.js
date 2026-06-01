#!/usr/bin/env node
/** End-to-end demo: convert MyDemo_CUDA and verify against MyDemo_UPTK. */

const fs = require("fs");
const path = require("path");
const { loadApiMap, convertSource, normalize } = require("./convert_lib");

const ROOT = path.resolve(__dirname, "..", "..");
const INPUT = path.join(ROOT, "example_port", "cuda_port_demo", "MyDemo_CUDA", "MyDemo.cu");
const EXPECTED = path.join(ROOT, "example_port", "cuda_port_demo", "MyDemo_UPTK", "MyDemo.cu");
const OUT_DIR = path.join(__dirname, "_test_output");

function main() {
  fs.rmSync(OUT_DIR, { recursive: true, force: true });
  fs.mkdirSync(OUT_DIR, { recursive: true });

  const outputFile = path.join(OUT_DIR, "MyDemo.cu");
  const apiMap = loadApiMap();
  const converted = convertSource(fs.readFileSync(INPUT, "utf8"), apiMap);
  fs.writeFileSync(outputFile, converted, "utf8");

  const actual = normalize(fs.readFileSync(outputFile, "utf8"));
  const expected = normalize(fs.readFileSync(EXPECTED, "utf8"));
  if (actual !== expected) {
    console.error("End-to-end demo conversion mismatch");
    process.exit(1);
  }

  console.log("PASS: end-to-end demo conversion");
  console.log(`Output written to ${outputFile}`);
}

main();
