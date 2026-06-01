/** Shared CUDA -> UPTK conversion helpers. */

const fs = require("fs");
const path = require("path");

const DEFAULT_MAP = path.join(__dirname, "api_map.json");

function loadApiMap(mapPath = DEFAULT_MAP) {
  return JSON.parse(fs.readFileSync(mapPath, "utf8"));
}

function protectIncludeLines(text) {
  const placeholders = {};
  const protectedText = text.replace(/^[ \t]*#include[^\n]*$/gm, (line) => {
    const key = `__UPTK_INCLUDE_${Object.keys(placeholders).length}__`;
    placeholders[key] = line;
    return key;
  });
  return {
    protectedText,
    restore(value) {
      let restored = value;
      for (const [key, original] of Object.entries(placeholders)) {
        restored = restored.replace(key, original);
      }
      return restored;
    },
  };
}

function applyMapping(text, mapping, fallback) {
  let out = text;
  for (const [sourceToken, targetToken] of Object.entries(mapping)) {
    if (sourceToken === targetToken) continue;
    const pattern = new RegExp(
      `\\b${sourceToken.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`,
      "g"
    );
    out = out.replace(pattern, targetToken);
  }
  if (fallback) out = out.replace(fallback.re, fallback.repl);
  return out;
}

function insertAfterCudaInclude(text, includeLine) {
  const lines = text.split(/(?<=\n)/);
  let insertAt = 0;
  for (let index = 0; index < lines.length; index++) {
    const line = lines[index];
    if (
      line.includes("#include") &&
      /cuda_runtime|cuda\.h|driver_types/.test(line)
    ) {
      insertAt = index + 1;
      break;
    }
  }
  if (text.includes(includeLine)) return text;
  if (insertAt < lines.length && lines[insertAt].trim() === "") {
    lines[insertAt] = `${includeLine}\n`;
  } else {
    lines.splice(insertAt, 0, `${includeLine}\n`);
  }
  return lines.join("");
}

function convertSource(source, apiMap) {
  const { protectedText, restore } = protectIncludeLines(source);
  let text = protectedText;

  text = applyMapping(text, apiMap.driver_functions, {
    re: /\bcu([A-Z][A-Za-z0-9_]*)\b/g,
    repl: (_, tail) => `UP${tail}`,
  });
  text = applyMapping(text, apiMap.runtime_functions, {
    re: /\bcuda([A-Z][A-Za-z0-9_]*)\b/g,
    repl: (_, tail) => `UPTK${tail}`,
  });
  text = applyMapping(text, apiMap.driver_types, {
    re: /\bCU([A-Z][A-Za-z0-9_]*)\b/g,
    repl: (_, tail) => `UPTK${tail}`,
  });
  text = applyMapping(text, apiMap.runtime_types, {
    re: /\bcuda([A-Z][A-Za-z0-9_]*)\b/g,
    repl: (_, tail) => `UPTK${tail}`,
  });

  text = restore(text);
  if (
    text.includes("UPTKMalloc") ||
    text.includes("UPTKMemcpy") ||
    text.includes("UPTKDeviceSynchronize")
  ) {
    text = insertAfterCudaInclude(text, "#include <UPTK_runtime_api.h>");
  }
  if (text.includes("UPInit") || text.includes("UPCtxCreate")) {
    text = insertAfterCudaInclude(text, "#include <UPTK.h>");
  }
  return text;
}

function normalize(text) {
  return `${text.replace(/\r\n/g, "\n").split("\n").map((l) => l.replace(/\s+$/, "")).join("\n").trim()}\n`;
}

module.exports = {
  loadApiMap,
  convertSource,
  normalize,
};
