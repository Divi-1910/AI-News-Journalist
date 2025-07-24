import React, { useState } from "react";
import { useAtom } from "jotai";
import {
  Mic,
  Send,
  UserCircle,
  Settings,
  LogOut,
  Bot,
  Sun,
  Moon,
  ChevronsLeft,
  ChevronsRight
} from "lucide-react";
import { userAtom } from "../store/AuthStore";
import { AuthApi } from "../api/api";

const Chat = () => {
  const [user] = useAtom(userAtom);
  const [messages, setMessages] = useState([
    {
      sender: "Anya",
      text: "Welcome! What news can I get for you today?",
      personality: user?.preferences?.news_personality || "analytical_analyst"
    }
  ]);
  const [input, setInput] = useState("");
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  const handleLogout = () => {
    AuthApi.logout();
    window.location.href = "/";
  };

  const handleSendMessage = (e) => {
    e.preventDefault();
    if (input.trim() === "") return;

    // Add user message
    setMessages([...messages, { sender: "user", text: input }]);
    setInput("");

    // TODO: Add logic to get Anya's response
  };

  const personalityMap = {
    analytical_analyst: {
      name: "Analytical Analyst",
      color: "text-blue-400",
      icon: <Bot className="w-8 h-8 text-blue-400" />
    },
    charismatic_anchor: {
      name: "Charismatic Anchor",
      color: "text-purple-400",
      icon: <Bot className="w-8 h-8 text-purple-400" />
    },
    seasoned_journalist: {
      name: "Seasoned Journalist",
      color: "text-amber-400",
      icon: <Bot className="w-8 h-8 text-amber-400" />
    },
    curious_explorer: {
      name: "Curious Explorer",
      color: "text-emerald-400",
      icon: <Bot className="w-8 h-8 text-emerald-400" />
    },
    witty_intern: {
      name: "Witty Intern",
      color: "text-rose-400",
      icon: <Bot className="w-8 h-8 text-rose-400" />
    }
  };

  const currentPersonality =
    personalityMap[user?.preferences?.news_personality] ||
    personalityMap["analytical_analyst"];

  return (
    <div className="flex h-screen bg-slate-900 text-white overflow-hidden">
      {/* Sidebar */}
      <aside
        className={`bg-slate-800/50 backdrop-blur-sm border-r border-white/10 transition-all duration-300 ${
          isSidebarOpen ? "w-64" : "w-20"
        } flex flex-col`}
      >
        <div className="p-4 border-b border-white/10 flex items-center justify-between">
          {isSidebarOpen && <h1 className="text-xl font-bold">Anya</h1>}
          <button
            onClick={() => setIsSidebarOpen(!isSidebarOpen)}
            className="p-2 hover:bg-white/10 rounded-lg"
          >
            {isSidebarOpen ? <ChevronsLeft /> : <ChevronsRight />}
          </button>
        </div>

        <nav className="flex-1 p-4 space-y-2">
          {/* Add chat history items here */}
        </nav>

        <div className="p-4 border-t border-white/10">
          <div className="flex items-center space-x-3 mb-4">
            <UserCircle className="w-10 h-10 text-gray-400" />
            {isSidebarOpen && (
              <div>
                <p className="font-semibold">{user?.profile?.name}</p>
                <p className="text-xs text-gray-400">{user?.profile?.email}</p>
              </div>
            )}
          </div>
          <button
            onClick={handleLogout}
            className="w-full flex items-center p-2 bg-rose-500/20 hover:bg-rose-500/40 rounded-lg"
          >
            <LogOut className="w-5 h-5 mr-3" />
            {isSidebarOpen && "Logout"}
          </button>
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col">
        {/* Header */}
        <header className="bg-white/5 backdrop-blur-sm border-b border-white/10 p-4 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            {currentPersonality.icon}
            <div>
              <h2 className="text-lg font-semibold">Anya</h2>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse"></div>
                <p className="text-xs text-gray-400">Online</p>
              </div>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <button className="p-2 hover:bg-white/10 rounded-lg">
              <Sun className="w-5 h-5" />
            </button>
            <button className="p-2 hover:bg-white/10 rounded-lg">
              <Settings className="w-5 h-5" />
            </button>
          </div>
        </header>

        {/* Message Display */}
        <div className="flex-1 p-6 overflow-y-auto">
          <div className="max-w-4xl mx-auto space-y-6">
            {messages.map((msg, index) => (
              <div
                key={index}
                className={`flex items-start gap-4 ${
                  msg.sender === "user" ? "justify-end" : ""
                }`}
              >
                {msg.sender === "Anya" && (
                  <div className="w-8 h-8 rounded-full bg-gradient-to-r from-purple-500 to-pink-500 flex-shrink-0 flex items-center justify-center font-bold">
                    A
                  </div>
                )}
                <div
                  className={`max-w-lg p-4 rounded-2xl ${
                    msg.sender === "user"
                      ? "bg-purple-600 rounded-br-none"
                      : "bg-slate-700 rounded-bl-none"
                  }`}
                >
                  <p className="text-white">{msg.text}</p>
                </div>
                {msg.sender === "user" && (
                  <UserCircle className="w-8 h-8 text-gray-400 flex-shrink-0" />
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Message Input */}
        <div className="p-6 bg-slate-800/50 border-t border-white/10">
          <form
            onSubmit={handleSendMessage}
            className="max-w-4xl mx-auto flex items-center bg-slate-700/50 rounded-xl p-2 border border-transparent focus-within:border-purple-500"
          >
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask Anya about the latest news..."
              className="flex-1 bg-transparent px-4 py-2 text-white placeholder-gray-400 focus:outline-none"
            />
            <button
              type="button"
              className="p-3 text-gray-400 hover:text-white"
            >
              <Mic className="w-5 h-5" />
            </button>
            <button
              type="submit"
              className="p-3 bg-purple-600 rounded-lg hover:bg-purple-500 disabled:bg-gray-600 disabled:cursor-not-allowed"
              disabled={!input.trim()}
            >
              <Send className="w-5 h-5 text-white" />
            </button>
          </form>
        </div>
      </main>
    </div>
  );
};

export default Chat;
