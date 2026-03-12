import express from "express";
import { pipeline } from "@xenova/transformers";

const app = express();
app.use(express.json());

let extractor;

// load model once
async function loadModel() {
  console.log("Loading embedding model...");
  extractor = await pipeline(
    "feature-extraction",
    "Xenova/all-MiniLM-L6-v2"
  );
  console.log("Model loaded");
}

await loadModel();

app.post("/embed", async (req, res) => {
  try {
    const { text } = req.body;

    const output = await extractor(text, {
      pooling: "mean",
      normalize: true,
    });

    res.json({
      embedding: Array.from(output.data),
    });

  } catch (error) {
    console.error(error);
    res.status(500).send("Embedding failed");
  }
});

app.listen(3001, () => {
  console.log("Embedding server running on port 3001");
});