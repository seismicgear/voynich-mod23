/*
 * Voynich Workbench — Electron shell.
 *
 * Spawns the PyInstaller backend (bundled under resources/backend),
 * waits for it to announce its URL, and opens it in a native window.
 * `--smoke` starts everything, verifies the page title, prints SMOKE OK
 * and exits — used by CI on all three platforms.
 */

const { app, BrowserWindow, dialog } = require("electron");
const { spawn } = require("child_process");
const path = require("path");

const SMOKE = process.argv.includes("--smoke");
const STARTUP_TIMEOUT_MS = 90_000;

let backend = null;
let quitting = false;

function backendPath() {
  const name =
    process.platform === "win32" ? "VoynichWorkbench.exe" : "VoynichWorkbench";
  if (app.isPackaged) {
    return path.join(process.resourcesPath, "backend", name);
  }
  return path.join(__dirname, "..", "dist", name);
}

function startBackend() {
  return new Promise((resolve, reject) => {
    backend = spawn(backendPath(), ["--no-browser", "--port", "0"], {
      env: { ...process.env },
    });
    const timer = setTimeout(
      () => reject(new Error("backend did not start within 90s")),
      STARTUP_TIMEOUT_MS
    );
    let buf = "";
    const onData = (chunk) => {
      buf += chunk.toString();
      const m = buf.match(/Serving at (http:\/\/127\.0\.0\.1:\d+)\//);
      if (m) {
        clearTimeout(timer);
        resolve(m[1]);
      }
    };
    backend.stdout.on("data", onData);
    backend.stderr.on("data", onData);
    backend.on("error", (err) => {
      clearTimeout(timer);
      reject(err);
    });
    backend.on("exit", (code) => {
      if (!quitting) {
        clearTimeout(timer);
        reject(new Error(`backend exited early (code ${code})\n${buf.slice(-800)}`));
      }
    });
  });
}

const SPLASH = `data:text/html,${encodeURIComponent(`<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Voynich Workbench</title></head>
<body style="margin:0;display:flex;align-items:center;justify-content:center;
height:100vh;background:#e3d5b3;color:#8a3324;
font:small-caps 600 22px Georgia,serif;letter-spacing:1px">
Unrolling the manuscript&hellip;</body></html>`)}`;

async function createWindow() {
  const win = new BrowserWindow({
    width: 1280,
    height: 900,
    backgroundColor: "#e3d5b3",
    title: "Voynich Workbench",
    show: !SMOKE,
    webPreferences: { contextIsolation: true, nodeIntegration: false },
  });
  win.setMenuBarVisibility(false);
  await win.loadURL(SPLASH);

  try {
    const url = await startBackend();
    await win.loadURL(url);
    if (SMOKE) {
      const title = await win.webContents.executeJavaScript("document.title");
      if (String(title).includes("Voynich")) {
        console.log("SMOKE OK");
        app.exit(0);
      } else {
        console.error(`SMOKE FAILED: unexpected title "${title}"`);
        app.exit(1);
      }
    }
  } catch (err) {
    if (SMOKE) {
      console.error(`SMOKE FAILED: ${err.message}`);
      app.exit(1);
    } else {
      dialog.showErrorBox("Voynich Workbench failed to start", String(err.message));
      app.quit();
    }
  }
}

const gotLock = app.requestSingleInstanceLock();
if (!gotLock) {
  app.quit();
} else {
  app.whenReady().then(createWindow);
}

app.on("window-all-closed", () => app.quit());

app.on("will-quit", () => {
  quitting = true;
  if (backend && !backend.killed) {
    backend.kill();
  }
});
