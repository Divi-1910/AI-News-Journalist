import React, { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { useAtom } from "jotai";
import {
  Send,
  Mic,
  Settings,
  LogOut,
  User,
  Bot,
  Sparkles,
  Clock,
  ExternalLink,
  Copy,
  ThumbsUp,
  ThumbsDown
} from "lucide-react";
import { userAtom } from "../store/AuthStore";
import { AuthApi } from "../api/api";
import SettingsDialog from "../components/SettingsDialog";

const useAutosizeTextArea = (textAreaRef, value) => {
  useEffect(() => {
    if (textAreaRef) {
      textAreaRef.style.height = "0px";
      const scrollHeight = textAreaRef.scrollHeight;
      textAreaRef.style.height = scrollHeight + "px";
    }
  }, [textAreaRef, value]);
};

const Chat = () => {
  const [user] = useAtom(userAtom);
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);
  const textAreaRef = useRef(null);
  const [showSettings, setShowSettings] = useState(false);
  const navigate = useNavigate();

  useAutosizeTextArea(textAreaRef.current, inputValue);

  // Initial welcome message
  useEffect(() => {
    const welcomeMessage = {
      id: "welcome",
      type: "bot",
      content: `Hello ${
        user?.profile?.name || "there"
      }! I'm Anya, your personalized AI news anchor. Ask me about any topic, and I'll fetch the latest information, tailored to your chosen personality. 📰✨`,
      timestamp: new Date(),
      sources: []
    };
    setMessages([welcomeMessage]);

    // We need to work here
    // to start the conversation
  }, [user]);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isTyping]);

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage = {
      id: Date.now().toString(),
      type: "user",
      content: inputValue,
      timestamp: new Date()
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue("");
    setIsLoading(true);
    setIsTyping(true);

    // Mock API call
    setTimeout(() => {
      const botResponse = {
        id: (Date.now() + 1).toString(),
        type: "bot",
        content: `Great question about "${inputValue}"! I'm analyzing the latest news sources to provide you with a comprehensive summary in the style of a **${
          user?.preferences?.news_personality?.replace(/_/g, " ") || "Analyst"
        }**. This is a sample response to demonstrate the UI.`,
        timestamp: new Date(),
        sources: [
          { title: "Reuters", url: "#" },
          { title: "Associated Press", url: "#" },
          { title: "BBC News", url: "#" }
        ]
      };

      // we will work here later to get the responses from the workflow manager (ai service)

      setIsTyping(false);
      setMessages((prev) => [...prev, botResponse]);
      setIsLoading(false);
    }, 2500);
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleLogout = () => {
    AuthApi.logout();
    navigate("/"); // Redirect to home after logout
  };

  const formatTimestamp = (timestamp) => {
    return new Intl.DateTimeFormat("en-US", {
      hour: "2-digit",
      minute: "2-digit",
      hour12: true
    }).format(timestamp);
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text).then(() => {
      // You can add a toast notification here to confirm copy
      alert("Copied to clipboard!");
    });
  };

  return (
    <div className="h-screen w-screen bg-slate-900 flex flex-col">
      {/* Animated Background */}
      <div className="absolute inset-0 overflow-hidden z-0">
        <div className="absolute top-0 -left-4 w-72 h-72 bg-purple-600/50 rounded-full mix-blend-multiply filter blur-xl opacity-50 animate-blob"></div>
        <div className="absolute -bottom-8 right-20 w-72 h-72 bg-blue-600/50 rounded-full mix-blend-multiply filter blur-xl opacity-50 animate-blob animation-delay-4000"></div>
      </div>

      {/* Header */}
      <header className="relative z-10 bg-slate-900/70 backdrop-blur-sm border-b border-white/10 p-4">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="relative">
              <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center">
                <Bot className="w-7 h-7 text-white" />
              </div>
              <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-green-400 rounded-full border-2 border-slate-900" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">Anya</h1>
              <p className="text-sm text-purple-300">
                {user?.preferences?.news_personality
                  ?.replace(/_/g, " ")
                  .replace(/\b\w/g, (l) => l.toUpperCase()) ||
                  "AI News Assistant"}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <div className="hidden md:flex items-center gap-2 bg-white/10 rounded-lg px-3 py-2">
              <User className="w-4 h-4 text-gray-300" />
              <span className="text-sm font-medium text-white">
                {user?.profile?.name}
              </span>
            </div>
            <button
              onClick={() => setShowSettings(true)}
              className="p-2 text-gray-300 hover:text-white hover:bg-white/20 rounded-lg transition-colors"
            >
              <Settings className="w-5 h-5" />
            </button>
            <button
              onClick={handleLogout}
              className="p-2 text-gray-300 hover:text-white hover:bg-white/20 rounded-lg transition-colors"
            >
              <LogOut className="w-5 h-5" />
            </button>
          </div>
        </div>
      </header>

      {/* Chat Messages Area */}
      <main className="flex-1 overflow-y-auto p-4 custom-scrollbar">
        <div className="max-w-4xl mx-auto space-y-8">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex items-start gap-4 animate-fade-in-up ${
                message.type === "user" ? "justify-end" : ""
              }`}
            >
              {message.type === "bot" && (
                <div className="w-10 h-10 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center flex-shrink-0">
                  <Bot className="w-5 h-5 text-white" />
                </div>
              )}
              <div
                className={`flex flex-col max-w-2xl ${
                  message.type === "user" ? "items-end" : "items-start"
                }`}
              >
                <div
                  className={`p-4 rounded-t-2xl ${
                    message.type === "user"
                      ? "bg-gradient-to-br from-purple-600 to-pink-600 text-white rounded-l-2xl"
                      : "bg-slate-800/80 text-gray-200 rounded-r-2xl"
                  }`}
                >
                  <p className="text-base leading-relaxed whitespace-pre-wrap">
                    {message.content}
                  </p>
                  {message.type === "bot" && message.sources?.length > 0 && (
                    <div className="mt-4 pt-3 border-t border-white/10">
                      <p className="text-xs font-semibold text-gray-400 mb-2">
                        SOURCES
                      </p>
                      <div className="flex flex-wrap gap-2">
                        {message.sources.map((source, index) => (
                          <a
                            key={index}
                            href={source.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center gap-1.5 bg-white/10 hover:bg-white/20 rounded-md px-2 py-1 text-xs text-gray-300 transition-colors"
                          >
                            <ExternalLink className="w-3 h-3" />
                            {source.title}
                          </a>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
                <div className="flex items-center gap-3 mt-2 px-1">
                  <span className="text-xs text-gray-500">
                    {formatTimestamp(message.timestamp)}
                  </span>
                  {message.type === "bot" && (
                    <div className="flex items-center gap-1 text-gray-400">
                      <button
                        onClick={() => copyToClipboard(message.content)}
                        className="p-1 hover:bg-white/10 rounded-full transition-colors"
                      >
                        <Copy className="w-3.5 h-3.5" />
                      </button>
                      <button className="p-1 hover:bg-white/10 rounded-full transition-colors">
                        <ThumbsUp className="w-3.5 h-3.5" />
                      </button>
                      <button className="p-1 hover:bg-white/10 rounded-full transition-colors">
                        <ThumbsDown className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  )}
                </div>
              </div>
              {message.type === "user" && (
                <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-full flex items-center justify-center flex-shrink-0">
                  <User className="w-5 h-5 text-white" />
                </div>
              )}
            </div>
          ))}

          {isTyping && (
            <div className="flex items-start gap-4 animate-fade-in-up">
              <div className="w-10 h-10 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center flex-shrink-0">
                <Bot className="w-5 h-5 text-white" />
              </div>
              <div className="bg-slate-800/80 rounded-2xl p-4">
                <div className="flex items-center space-x-1.5">
                  <span className="h-2 w-2 bg-purple-400 rounded-full animate-bounce delay-0" />
                  <span className="h-2 w-2 bg-purple-400 rounded-full animate-bounce delay-150" />
                  <span className="h-2 w-2 bg-purple-400 rounded-full animate-bounce delay-300" />
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </main>

      {/* Input Area */}
      <footer className="relative z-10 bg-slate-900/70 backdrop-blur-sm border-t border-white/10 p-4">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-end gap-3 bg-slate-800/80 rounded-2xl border border-white/10 p-2">
            <div className="flex-1 pl-2">
              <textarea
                ref={textAreaRef}
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask Anya anything..."
                className="w-full bg-transparent text-white placeholder-gray-400 resize-none focus:outline-none max-h-40 custom-scrollbar text-base"
                rows="1"
                disabled={isLoading}
              />
            </div>
            <div className="flex items-center gap-1">
              <button className="p-2 text-gray-400 hover:text-white transition-colors rounded-full hover:bg-white/10">
                <Mic className="w-5 h-5" />
              </button>
              <button
                onClick={handleSendMessage}
                disabled={!inputValue.trim() || isLoading}
                className="p-3 rounded-xl transition-all duration-300 transform flex items-center justify-center
                           disabled:bg-gray-600 disabled:text-gray-400 disabled:cursor-not-allowed
                           enabled:bg-gradient-to-r enabled:from-purple-500 enabled:to-pink-500 enabled:text-white enabled:hover:scale-110"
              >
                {isLoading ? (
                  <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                ) : (
                  <Send className="w-5 h-5" />
                )}
              </button>
            </div>
          </div>
          <p className="text-xs text-gray-500 text-center mt-3">
            Anya can make mistakes. Always verify important information.
          </p>
        </div>
      </footer>
      <SettingsDialog
        isOpen={showSettings}
        onClose={() => setShowSettings(false)}
      />
    </div>
  );
};

export default Chat;
