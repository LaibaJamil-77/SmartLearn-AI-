//import logo from './logo.svg';
import React from "react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import AppLayout from "./components/common/heading/AppLayout";
import ProcessingView from "./components/common/heading/ProcessingView";
import ChatPage from "./components/common/heading/ChatPage";
import HistoryPage from "./components/common/heading/HistoryPage";
import InputSelection from "./components/common/heading/InputSelection";
import Head from "./components/common/heading/Head";
import Header from './components/common/heading/Header';
import './App.css';

// Placeholder components for steps
const Step1 = () => <div style={{ padding: 40, fontSize: 28 }}>Step 1 Content</div>;
const Step2 = () => <div style={{ padding: 40, fontSize: 28 }}>Step 2 Content</div>;

function App() {
  return (
    <Router>
      <Header />
      <Routes>
        <Route element={<AppLayout />}>
          <Route path="/step1" element={<Step1 />} />
          <Route path="/step2" element={<Step2 />} />
          <Route path="/processing" element={<ProcessingView />} />
          <Route path="/chat" element={<ChatPage />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/input-selection" element={<InputSelection />} />
        </Route>
        <Route path="/" element={<Head />} />
      </Routes>
    </Router>
  );
}

export default App;
