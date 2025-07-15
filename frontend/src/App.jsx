import { useState } from "react";
import axios from "axios";

function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");

  const handleAsk = async () => {
    try {
      const res = await axios.post("http://localhost:8000/ask", {
        question,
      });
      setAnswer(res.data.answer);
    } catch {
      setAnswer("เกิดข้อผิดพลาด");
    }
  };

  return (
    <div style={{ padding: 20, maxWidth: 600, margin: "auto" }}>
      <h1>GraphRAG Q&A</h1>
      <input
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        style={{ width: "100%", padding: 8 }}
        placeholder="พิมพ์คำถาม"
      />
      <button onClick={handleAsk} style={{ marginTop: 10 }}>
        ถาม
      </button>
      <div style={{ marginTop: 20 }}>
        <strong>คำตอบ:</strong>
        <p>{answer}</p>
      </div>
    </div>
  );
}

export default App;
