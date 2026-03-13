const express = require('express')
const { pipeline } = require('@xenova/transformers')

const app = express();
app.use(express.json());

let featureExtraction;

const createFeatureExtraction = async () => {
    return pipeline(
    "feature-extraction",
    "Xenova/all-MiniLM-L6-v2"
  )
};

// load model once
async function getFeatureExtraction() {
    if (!featureExtraction) {
        // Assign the promise itself to the variable
        featureExtraction = createFeatureExtraction();
    }
    console.log('featureExtraction', featureExtraction)
    // Await the promise, all callers will await the same promise
    return featureExtraction;
}

app.post("/embed", async (req, res) => {
  try {
    const { text } = req.body;

    const embeddingModel = await getFeatureExtraction()
    console.log('embeddingModel', embeddingModel)

    const output = await embeddingModel(text, {
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