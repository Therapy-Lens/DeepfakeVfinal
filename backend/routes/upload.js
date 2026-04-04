const express = require("express");
const multer = require("multer");
const path = require("path");

const router = express.Router();

// storage config
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, path.join(__dirname, "../.uploads"));
  },
  filename: (req, file, cb) => {
    const uniqueName = Date.now() + path.extname(file.originalname);
    cb(null, uniqueName);
  },
});

const upload = multer({ storage });

const { execFile } = require("child_process");
const fs = require("fs");

router.post("/upload", upload.array("files", 10), async (req, res) => {
  try {
    const files = req.files;

    if (!files || files.length === 0) {
      return res.status(400).json({ error: "No files uploaded" });
    }

    console.log("Files received:", files.length);

    const filePath = files[0].path;
    const isVideo = files[0].mimetype.startsWith('video');
    const isAudio = files[0].mimetype.startsWith('audio');
    
    const pythonPath = path.join(__dirname, "../../venv/Scripts/python.exe");
    
    if (isAudio) {
        console.log("Detected AUDIO file - Saved to .uploads/ directory.");
        // Returns generic status without triggering python or deleting the file from disk!
        return res.json({ prediction: "UNCERTAIN", confidence: 0, status: "Audio file saved for development." });
    }
    
    let scriptPath;
    if (isVideo) {
        scriptPath = path.join(__dirname, "../utils/predict_video.py");
    } else {
        scriptPath = path.join(__dirname, "../utils/predict.py");
    }

    const { spawn } = require("child_process");
    
    console.log("Spawning Python Process...");
    const pythonProcess = spawn(pythonPath, [scriptPath, filePath]);
    
    let stdoutData = "";
    let stderrData = "";

    // 1 & 4. Handle multiple chunks logically concatenating stdout buffering
    pythonProcess.stdout.on("data", (data) => {
        stdoutData += data.toString();
    });

    pythonProcess.stderr.on("data", (data) => {
        stderrData += data.toString();
    });

    pythonProcess.on("error", (err) => {
        console.error("Critical Spawn Error:", err.message);
        if (!res.headersSent) {
             return res.status(500).json({ error: "Failed to execute python pipeline" });
        }
    });

    // 3. Ensure Response is Sent AFTER execution closes strictly
    pythonProcess.on("close", (code) => {
        console.log(`Python script finished execution with code ${code}.`);
        if (stderrData) console.log("STDERR WARNINGS:", stderrData);

        // 5. Handle Errors natively without breaking if JSON was printed
        if (code !== 0 && !stdoutData.trim()) {
            console.error("Execution Error. Python script exited with code:", code);
            return res.status(500).json({ error: "Prediction process failed" });
        }

        try {
            console.log("Processing Python Output Buffer...");
            
            // 2. SEARCH FOR JSON BLOCK (Bulletproof Scanning)
            // Even if python prints warnings BEFORE or AFTER the JSON, we find it.
            const jsonMatch = stdoutData.match(/\{"prediction":.*?\}/);
            
            if (!jsonMatch) {
                throw new Error("No valid JSON prediction found in script output");
            }

            const result = JSON.parse(jsonMatch[0]);
            console.log("Final Validated Result:", result);
            
            if (!res.headersSent) {
                res.json(result);
            }
            
            // 7. ADD DEBUG LOGS
            console.log("DEBUG: Response sent to frontend successfully.");
            console.log("DEBUG: Scheduling safe cleanup in 500ms...");

            // SAFE ASYNC CLEANUP WITH DELAY (Windows Lock Protection)
            setTimeout(() => {
                files.forEach(f => {
                    if (fs.existsSync(f.path)) {
                        fs.unlink(f.path, (err) => {
                            if (err) {
                                console.error(`[CLEANUP ERROR] Failed to delete ${f.path}:`, err.message);
                            } else {
                                console.log(`[CLEANUP] Deleted file: ${f.path}`);
                            }
                        });
                    }
                });
            }, 500);
            
        } catch (parseError) {
            console.error("Result Extraction Failed. Raw Output was:", stdoutData);
            if (!res.headersSent) {
                res.status(500).json({ error: "ML Engine failed to produce a valid response" });
            }
        }
    });

  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Upload failed" });
  }
});

module.exports = router;
