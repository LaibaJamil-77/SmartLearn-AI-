import React, { useEffect, useState, useRef } from "react";
import { FaUserCircle, FaRobot, FaTimes } from "react-icons/fa";

const HistoryPage = ({ onReAsk, onClose }) => {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const chatRefs = useRef([]);

  useEffect(() => {
    fetch("https://fyp-chatbot-vz6d.onrender.com/qa-history", {
      headers: {
        "Authorization": `Bearer ${localStorage.getItem("token")}`,
      },
    })
      .then(res => res.json())
      .then(data => {
        setHistory(data);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  // Scroll to chat when sidebar item is clicked
  const scrollToChat = idx => {
    if (chatRefs.current[idx]) {
      chatRefs.current[idx].scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  return (
    <div style={{
      display: 'flex',
      width: '900px',
      maxWidth: '98vw',
      height: '80vh',
      background: '#22232a',
      borderRadius: 16,
      boxShadow: '0 4px 24px rgba(0,0,0,0.10)',
      overflow: 'hidden',
      position: 'relative',
      color: '#fff',
    }}>
      {/* Sidebar */}
      <div style={{
        width: 260,
        background: '#17181c',
        color: '#fff',
        display: 'flex',
        flexDirection: 'column',
        borderRight: '1.5px solid #23242a',
      }}>
        <div style={{ fontWeight: 700, fontSize: 18, padding: '1.5rem 1.5rem 1rem 1.5rem', letterSpacing: 1 }}>Chats</div>
        <div style={{ flex: 1, overflowY: 'auto', padding: '0 1.5rem 1.5rem 1.5rem' }}>
          {loading ? (
            <div style={{ color: '#aaa', marginTop: 40 }}>Loading...</div>
          ) : history.length === 0 ? (
            <div style={{ color: '#aaa', marginTop: 40 }}>No chat history found.</div>
          ) : (
            history.map((item, idx) => (
              <div
                key={idx}
                style={{
                  padding: '10px 0',
                  borderBottom: '1px solid #23242a',
                  cursor: 'pointer',
                  color: '#eee',
                  fontSize: 15,
                  whiteSpace: 'nowrap',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  transition: 'background 0.15s',
                }}
                onClick={() => scrollToChat(idx)}
                title={item.question}
              >
                {item.question}
              </div>
            ))
          )}
        </div>
      </div>
      {/* Main chat area */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', position: 'relative', background: '#22232a' }}>
        {/* Close button */}
        <button
          style={{
            position: 'absolute',
            top: 18,
            right: 18,
            background: '#22232a',
            color: '#fff',
            border: 'none',
            borderRadius: '50%',
            width: 38,
            height: 38,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer',
            fontSize: 20,
            zIndex: 201,
          }}
          onClick={onClose}
          aria-label="Close chat history"
        >
          <FaTimes />
        </button>
        <div style={{
          fontSize: 22,
          fontWeight: 700,
          padding: '2rem 2rem 1rem 2rem',
          letterSpacing: '-1px',
          borderBottom: '1.5px solid #23242a',
        }}>
          Chat History
        </div>
        <div style={{
          flex: 1,
          overflowY: 'auto',
          padding: '2rem 2rem 2rem 2rem',
          background: '#22232a',
        }}>
          {loading ? (
            <div style={{ color: '#aaa', textAlign: 'center', marginTop: 40 }}>Loading chat history...</div>
          ) : history.length === 0 ? (
            <div style={{ color: '#aaa', textAlign: 'center', marginTop: 40 }}>No chat history found.</div>
          ) : (
            history.map((item, idx) => (
              <div
                key={idx}
                ref={el => chatRefs.current[idx] = el}
                style={{ marginBottom: 32, display: 'flex', flexDirection: 'column', gap: 8 }}
              >
                {/* User question bubble (right) */}
                <div style={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'center', gap: 8 }}>
                  <span style={{ color: '#2563eb', fontSize: 22 }}><FaUserCircle /></span>
                  <div
                    style={{
                      background: '#2563eb',
                      color: '#fff',
                      borderRadius: '18px 18px 4px 18px',
                      padding: '12px 20px',
                      maxWidth: 420,
                      fontSize: 16,
                      fontWeight: 500,
                      boxShadow: '0 2px 8px #2563eb22',
                      cursor: onReAsk ? 'pointer' : 'default',
                      transition: 'background 0.2s',
                    }}
                    title={onReAsk ? 'Click to re-ask this question' : ''}
                    onClick={() => onReAsk && onReAsk(item.question)}
                  >
                    {item.question}
                    <div style={{ fontSize: 12, color: '#e3eafc', marginTop: 4, textAlign: 'right' }}>
                      {item.asked_at ? new Date(item.asked_at).toLocaleString() : ''}
                    </div>
                  </div>
                </div>
                {/* Bot answer bubble (left) */}
                <div style={{ display: 'flex', justifyContent: 'flex-start', alignItems: 'center', gap: 8 }}>
                  <span style={{ color: '#7c3aed', fontSize: 22 }}><FaRobot /></span>
                  <div style={{
                    background: '#fff',
                    color: '#222',
                    borderRadius: '18px 18px 18px 4px',
                    padding: '12px 20px',
                    maxWidth: 480,
                    fontSize: 16,
                    fontWeight: 400,
                    boxShadow: '0 2px 8px #7c3aed22',
                    border: '1.5px solid #e3eafc',
                  }}>
                    {item.answer}
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

export default HistoryPage; 