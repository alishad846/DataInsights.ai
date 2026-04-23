import request from "supertest";
import mongoose from "mongoose";
import app from "../../src/index.js";
import { pipelineQueue } from "../../src/queue/pipelineQueue.js";

// Mock the queue so tests don't actually trigger Redis
jest.mock("../../src/queue/pipelineQueue.js", () => ({
  pipelineQueue: { add: jest.fn() }
}));

describe("Node.js Integration Tests", () => {

  beforeAll(async () => {
    // connect to local test DB or bypass
    process.env.JWT_SECRET = "test_secret";
  });

  afterAll(async () => {
    await mongoose.connection.close();
  });

  describe("GET /api/health", () => {
    it("should return healthy status", async () => {
      const res = await request(app).get("/api/health");
      expect(res.statusCode).toBe(200);
      expect(res.body).toHaveProperty("server", "running");
    });
  });

  describe("POST /api/upload/upload", () => {
    it("should reject unauthenticated upload requests", async () => {
      const res = await request(app)
        .post("/api/upload/upload")
        .attach("file", Buffer.from("Date,Sales,Profit\n2023-01-01,100,20"), "test.csv");

      expect(res.statusCode).toBe(401);
      expect(res.body.message).toMatch(/Not authorized/);
    });
  });

  describe("POST /api/chat/chat", () => {
    it("should reject payload without datasetId", async () => {
      const res = await request(app)
        .post("/api/chat/chat")
        .send({ message: "What is revenue?" });
        
      // Unauthorized or 400 depending on middleware
      expect([400, 401]).toContain(res.statusCode);
    });
  });
});
