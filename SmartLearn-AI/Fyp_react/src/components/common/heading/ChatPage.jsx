import React, { useState } from 'react';

const ChatPage = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');

  const handleSend = () => {
    if (!input.trim()) return;
    setMessages([...messages, { type: 'user', text: input }]);
    // Here you would call your backend or OpenAI API and add the response
    setTimeout(() => {
      setMessages(msgs => [...msgs, { type: 'bot', text: 'This is a sample response.' }]);
    }, 1000);
    setInput('');
  };

  return (
    <div className="chatgpt-ui" style={{ maxWidth: 700, margin: '40px auto', background: '#fff', borderRadius: 16, boxShadow: '0 4px 24px rgba(0,0,0,0.10)', padding: 32 }}>
      <div style={{ minHeight: 300, marginBottom: 24 }}>
        {messages.map((msg, idx) => (
          <div key={idx} style={{ textAlign: msg.type === 'user' ? 'right' : 'left', margin: '12px 0' }}>
            <span style={{
              display: 'inline-block',
              background: msg.type === 'user' ? '#4f8cff' : '#e3eafc',
              color: msg.type === 'user' ? '#fff' : '#222',
              borderRadius: 12,
              padding: '10px 18px',
              maxWidth: 400
            }}>
              {msg.text}
            </span>
          </div>
        ))}
      </div>
      <div style={{ display: 'flex', gap: 8 }}>
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleSend()}
          style={{ flex: 1, borderRadius: 8, border: '1px solid #ccc', padding: 12, fontSize: 16 }}
          placeholder="Type your message..."
        />
        <button onClick={handleSend} style={{ background: '#4f8cff', color: '#fff', border: 'none', borderRadius: 8, padding: '0 24px', fontSize: 16 }}>Send</button>
      </div>
    </div>
  );
};

export default ChatPage; 