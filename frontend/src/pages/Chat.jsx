import React, {
  useState,
  useEffect,
  useRef,
  useCallback,
  useMemo
} from "react";
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
  ThumbsDown,
  AlertCircle,
  CheckCircle,
  Loader,
  Trash2,
  ChevronDown,
  ChevronUp,
  Zap,
  Brain,
  Search,
  Filter,
  Globe,
  FileText,
  Palette,
  MessageSquare,
  Database,
  Share2,
  Bookmark,
  RotateCcw,
  Wifi,
  WifiOff,
  Volume2,
  VolumeX,
  Maximize2,
  Minimize2,
  MessageCircle,
  TrendingUp,
  Star,
  Eye,
  MoreHorizontal
} from "lucide-react";
import { userAtom } from "../store/AuthStore";
import { AuthApi, ChatApi } from "../api/api";
import SettingsDialog from "../components/SettingsDialog";
import WorkflowStats from "../components/WorkflowStats";
import { marked } from "marked";
import DOMPurify from "dompurify";

// Enhanced Agent Configuration
const AGENT_CONFIG = {
  classifier: {
    name: "Intent Analyzer",
    icon: Brain,
    color: "from-blue-500 to-cyan-500",
    textColor: "text-blue-400",
    description: "Understanding your question's intent",
    emoji: "🧠"
  },
  "keyword-extractor": {
    name: "Topic Extractor",
    icon: Search,
    color: "from-green-500 to-emerald-500",
    textColor: "text-green-400",
    description: "Identifying key topics and entities",
    emoji: "🔍"
  },
  "news-api": {
    name: "News Searcher",
    icon: Globe,
    color: "from-orange-500 to-red-500",
    textColor: "text-orange-400",
    description: "Searching global news sources",
    emoji: "🌍"
  },
  embedding: {
    name: "Content Processor",
    icon: Database,
    color: "from-purple-500 to-violet-500",
    textColor: "text-purple-400",
    description: "Converting content to vectors",
    emoji: "🔢"
  },
  relevancy: {
    name: "Relevance Filter",
    icon: Filter,
    color: "from-pink-500 to-rose-500",
    textColor: "text-pink-400",
    description: "Filtering most relevant articles",
    emoji: "⚡"
  },
  scraper: {
    name: "Content Gatherer",
    icon: FileText,
    color: "from-indigo-500 to-blue-500",
    textColor: "text-indigo-400",
    description: "Extracting full article content",
    emoji: "📄"
  },
  summarizer: {
    name: "Summary Generator",
    icon: Sparkles,
    color: "from-teal-500 to-cyan-500",
    textColor: "text-teal-400",
    description: "Creating personalized summaries",
    emoji: "✨"
  },
  persona: {
    name: "Style Formatter",
    icon: Palette,
    color: "from-yellow-500 to-orange-500",
    textColor: "text-yellow-400",
    description: "Applying your preferred style",
    emoji: "🎨"
  },
  memory: {
    name: "Context Manager",
    icon: Clock,
    color: "from-emerald-500 to-green-500",
    textColor: "text-emerald-400",
    description: "Managing conversation context",
    emoji: "💭"
  },
  chitchat: {
    name: "Conversation Handler",
    icon: MessageSquare,
    color: "from-rose-500 to-pink-500",
    textColor: "text-rose-400",
    description: "Handling casual conversation",
    emoji: "💬"
  }
};

// Custom Hooks
const useAutosizeTextArea = (ref, value) => {
  useEffect(() => {
    if (ref?.current) {
      ref.current.style.height = "0px";
      const scrollHeight = ref.current.scrollHeight;
      ref.current.style.height = `${Math.min(scrollHeight, 160)}px`;
    }
  }, [ref, value]);
};

const useToast = () => {
  const [toasts, setToasts] = useState([]);

  const addToast = useCallback((message, type = "info", duration = 3000) => {
    const id = Date.now();
    const toast = { id, message, type, duration };
    setToasts((prev) => [...prev, toast]);

    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, duration);
  }, []);

  const removeToast = useCallback((id) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  return { toasts, addToast, removeToast };
};

const useSound = () => {
  const [enabled, setEnabled] = useState(
    () => localStorage.getItem("Infiya_sound") !== "false"
  );

  const playSound = useCallback(
    (type) => {
      if (!enabled) return;

      // Create audio context for different sounds
      const audioContext = new (window.AudioContext ||
        window.webkitAudioContext)();
      const oscillator = audioContext.createOscillator();
      const gainNode = audioContext.createGain();

      oscillator.connect(gainNode);
      gainNode.connect(audioContext.destination);

      // Different sounds for different events
      switch (type) {
        case "send":
          oscillator.frequency.setValueAtTime(800, audioContext.currentTime);
          oscillator.frequency.exponentialRampToValueAtTime(
            400,
            audioContext.currentTime + 0.1
          );
          break;
        case "receive":
          oscillator.frequency.setValueAtTime(400, audioContext.currentTime);
          oscillator.frequency.exponentialRampToValueAtTime(
            600,
            audioContext.currentTime + 0.1
          );
          break;
        case "error":
          oscillator.frequency.setValueAtTime(200, audioContext.currentTime);
          break;
        default:
          oscillator.frequency.setValueAtTime(500, audioContext.currentTime);
      }

      gainNode.gain.setValueAtTime(0.1, audioContext.currentTime);
      gainNode.gain.exponentialRampToValueAtTime(
        0.01,
        audioContext.currentTime + 0.1
      );

      oscillator.start();
      oscillator.stop(audioContext.currentTime + 0.1);
    },
    [enabled]
  );

  const toggleSound = useCallback(() => {
    const newEnabled = !enabled;
    setEnabled(newEnabled);
    localStorage.setItem("Infiya_sound", newEnabled.toString());
  }, [enabled]);

  return { enabled, playSound, toggleSound };
};

// Enhanced Components
const Toast = ({ toast, onRemove }) => (
  <div
    className={`mb-2 p-4 rounded-lg shadow-lg backdrop-blur-lg border transition-all duration-300 animate-slide-in-right ${
      toast.type === "error"
        ? "bg-red-900/80 border-red-500/50 text-red-100"
        : toast.type === "success"
        ? "bg-green-900/80 border-green-500/50 text-green-100"
        : "bg-blue-900/80 border-blue-500/50 text-blue-100"
    }`}
  >
    <div className="flex items-center justify-between">
      <span className="text-sm font-medium">{toast.message}</span>
      <button
        onClick={() => onRemove(toast.id)}
        className="ml-3 text-white hover:text-gray-300 transition-colors"
      >
        ×
      </button>
    </div>
  </div>
);

const ToastContainer = ({ toasts, onRemove }) => (
  <div className="fixed top-4 right-4 z-50 space-y-2">
    {toasts.map((toast) => (
      <Toast key={toast.id} toast={toast} onRemove={onRemove} />
    ))}
  </div>
);

const ConnectionStatus = ({ isConnected, onRetry }) => (
  <div
    className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium transition-all ${
      isConnected
        ? "bg-green-900/50 text-green-300 border border-green-500/30"
        : "bg-red-900/50 text-red-300 border border-red-500/30"
    }`}
  >
    {isConnected ? (
      <>
        <Wifi className="w-3 h-3" />
        <span>Connected</span>
      </>
    ) : (
      <>
        <WifiOff className="w-3 h-3" />
        <span>Disconnected</span>
        <button
          onClick={onRetry}
          className="ml-1 hover:text-red-100 transition-colors"
        >
          <RotateCcw className="w-3 h-3" />
        </button>
      </>
    )}
  </div>
);

const MessageActions = ({ message, onCopy, onShare, onBookmark, onReact }) => {
  const [showActions, setShowActions] = useState(false);
  const [copiedRecently, setCopiedRecently] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopiedRecently(true);
      onCopy?.();
      setTimeout(() => setCopiedRecently(false), 2000);
    } catch (error) {
      console.error("Failed to copy:", error);
    }
  };

  return (
    <div className="relative">
      <button
        onClick={() => setShowActions(!showActions)}
        className="p-1 hover:bg-white/10 rounded-full transition-colors opacity-0 group-hover:opacity-100"
      >
        <MoreHorizontal className="w-4 h-4 text-gray-400" />
      </button>

      {showActions && (
        <div className="absolute bottom-full right-0 mb-2 bg-slate-800 border border-white/10 rounded-lg shadow-xl p-2 flex items-center gap-1 animate-fade-in">
          <button
            onClick={handleCopy}
            className="p-2 hover:bg-white/10 rounded-lg transition-colors"
            title="Copy message"
          >
            {copiedRecently ? (
              <CheckCircle className="w-4 h-4 text-green-400" />
            ) : (
              <Copy className="w-4 h-4 text-gray-400 hover:text-white" />
            )}
          </button>

          <button
            onClick={() => onShare?.(message)}
            className="p-2 hover:bg-white/10 rounded-lg transition-colors"
            title="Share message"
          >
            <Share2 className="w-4 h-4 text-gray-400 hover:text-white" />
          </button>

          <button
            onClick={() => onBookmark?.(message)}
            className="p-2 hover:bg-white/10 rounded-lg transition-colors"
            title="Bookmark message"
          >
            <Bookmark className="w-4 h-4 text-gray-400 hover:text-white" />
          </button>

          <div className="w-px h-6 bg-white/10" />

          <button
            onClick={() => onReact?.(message.id, "like")}
            className="p-2 hover:bg-white/10 rounded-lg transition-colors"
            title="Like message"
          >
            <ThumbsUp className="w-4 h-4 text-gray-400 hover:text-green-400" />
          </button>

          <button
            onClick={() => onReact?.(message.id, "dislike")}
            className="p-2 hover:bg-white/10 rounded-lg transition-colors"
            title="Dislike message"
          >
            <ThumbsDown className="w-4 h-4 text-gray-400 hover:text-red-400" />
          </button>
        </div>
      )}
    </div>
  );
};

const EnhancedThinkingIndicator = ({ agentProgress, onExpand }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const completedCount = agentProgress.filter(
    (a) => a.status === "completed"
  ).length;
  const totalCount = agentProgress.length;

  // Use the highest progress value from agents for smoother animation
  const maxProgress =
    agentProgress.length > 0
      ? Math.max(...agentProgress.map((a) => a.progress || 0)) * 100
      : 0;
  const progress = maxProgress;

  return (
    <div className="flex items-start gap-4 animate-fade-in-up">
      <div className="w-10 h-10 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center flex-shrink-0 shadow-lg relative">
        <Bot className="w-5 h-5 text-white" />
        <div className="absolute inset-0 rounded-full bg-gradient-to-r from-purple-500 to-pink-500 animate-ping opacity-20"></div>
      </div>

      <div className="w-full max-w-2xl">
        <div className="bg-slate-800/90 backdrop-blur-lg rounded-3xl p-5 border border-white/10 shadow-xl">
          {/* Header */}
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-purple-400 animate-pulse" />
                <span className="text-white font-semibold">
                  Infiya is analyzing...
                </span>
              </div>
            </div>

            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="flex items-center gap-2 px-3 py-1 text-xs text-gray-400 hover:text-white transition-colors rounded-lg hover:bg-white/10"
            >
              <span>{isExpanded ? "Hide" : "Show"} Details</span>
              {isExpanded ? (
                <ChevronUp className="w-3 h-3" />
              ) : (
                <ChevronDown className="w-3 h-3" />
              )}
            </button>
          </div>

          {/* Progress Bar */}
          {totalCount > 0 && (
            <div className="mb-4">
              <div className="w-full bg-slate-700 rounded-full h-2 overflow-hidden">
                <div
                  className="h-2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full transition-all duration-1000 ease-out"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          )}

          {/* Thinking Animation */}
          <div className="flex items-center justify-center space-x-2 mb-4">
            <div
              className="w-2 h-2 bg-purple-400 rounded-full animate-bounce"
              style={{ animationDelay: "0ms" }}
            ></div>
            <div
              className="w-2 h-2 bg-pink-400 rounded-full animate-bounce"
              style={{ animationDelay: "150ms" }}
            ></div>
            <div
              className="w-2 h-2 bg-purple-400 rounded-full animate-bounce"
              style={{ animationDelay: "300ms" }}
            ></div>
          </div>

          {/* Agent Details */}
          {isExpanded && agentProgress.length > 0 && (
            <div className="space-y-3 animate-fade-in">
              {agentProgress.map((agent, index) => {
                const config = AGENT_CONFIG[agent.agent] || {
                  name: agent.agent,
                  icon: Bot,
                  textColor: "text-gray-400",
                  color: "from-gray-500 to-gray-600",
                  emoji: "🤖"
                };
                const Icon = config.icon;

                return (
                  <div
                    key={index}
                    className={`flex items-center gap-3 p-3 rounded-xl transition-all duration-300 ${
                      agent.status === "completed"
                        ? "bg-green-900/20 border border-green-500/30"
                        : agent.status === "processing"
                        ? "bg-blue-900/20 border border-blue-500/30"
                        : "bg-slate-700/50 border border-slate-600/30"
                    }`}
                  >
                    <div
                      className={`w-8 h-8 rounded-lg flex items-center justify-center bg-gradient-to-r ${config.color}`}
                    >
                      <Icon className="w-4 h-4 text-white" />
                    </div>

                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-medium text-white">
                          {config.name}
                        </span>
                        <span className="text-xs">{config.emoji}</span>
                      </div>
                      <p className="text-xs text-gray-400 truncate">
                        {agent.message}
                      </p>
                    </div>

                    <div className="flex items-center gap-2">
                      {agent.processingTime && (
                        <span className="text-xs text-gray-500 bg-white/10 px-2 py-1 rounded-full">
                          {agent.processingTime}ms
                        </span>
                      )}
                      {agent.status === "completed" ? (
                        <CheckCircle className="w-4 h-4 text-green-400" />
                      ) : agent.status === "processing" ? (
                        <Loader className="w-4 h-4 text-blue-400 animate-spin" />
                      ) : (
                        <Clock className="w-4 h-4 text-gray-400" />
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

const EnhancedMessageAnalysis = ({ analysis, messageId }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!analysis || analysis.length === 0) return null;

  const completedAgents = analysis.filter(
    (a) => a.status === "completed"
  ).length;
  const totalProcessingTime = analysis.reduce(
    (sum, a) => sum + (a.processingTime || 0),
    0
  );

  return (
    <div className="mt-6 pt-4 border-t border-white/10">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center justify-between w-full text-left p-3 rounded-xl bg-white/5 hover:bg-white/10 transition-all duration-200 group"
      >
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-gradient-to-r from-cyan-500 to-blue-500 rounded-lg flex items-center justify-center">
            <Eye className="w-4 h-4 text-white" />
          </div>
          <div>
            <div className="text-sm font-medium text-white">
              AI Analysis Details
            </div>
            <div className="text-xs text-gray-400">
              {completedAgents} agents • {totalProcessingTime}ms total
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <div className="w-6 h-6 bg-white/10 rounded-full flex items-center justify-center">
            <TrendingUp className="w-3 h-3 text-cyan-400" />
          </div>
          {isExpanded ? (
            <ChevronUp className="w-4 h-4 text-gray-400 group-hover:text-white transition-colors" />
          ) : (
            <ChevronDown className="w-4 h-4 text-gray-400 group-hover:text-white transition-colors" />
          )}
        </div>
      </button>

      {isExpanded && (
        <div className="mt-4 space-y-3 animate-fade-in">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {analysis.map((agent, index) => {
              const config = AGENT_CONFIG[agent.agent] || {
                name: agent.agent,
                icon: Bot,
                textColor: "text-gray-400",
                color: "from-gray-500 to-gray-600",
                description: "Processing...",
                emoji: "🤖"
              };
              const Icon = config.icon;

              return (
                <div
                  key={index}
                  className="bg-white/5 p-4 rounded-xl border border-white/10 hover:bg-white/10 transition-all duration-200"
                >
                  <div className="flex items-start gap-3">
                    <div
                      className={`w-10 h-10 rounded-lg flex items-center justify-center bg-gradient-to-r ${config.color} flex-shrink-0`}
                    >
                      <Icon className="w-5 h-5 text-white" />
                    </div>

                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <h4 className="font-medium text-white text-sm">
                          {config.name}
                        </h4>
                        <span className="text-sm">{config.emoji}</span>
                      </div>
                      <p className="text-xs text-gray-400 mb-2">
                        {config.description}
                      </p>
                      <p className="text-xs text-gray-300">{agent.message}</p>

                      {agent.data && (
                        <div className="mt-2 text-xs text-gray-500">
                          {Object.entries(agent.data)
                            .slice(0, 2)
                            .map(([key, value]) => (
                              <div key={key} className="flex justify-between">
                                <span>{key}:</span>
                                <span className="font-mono">
                                  {typeof value === "object"
                                    ? JSON.stringify(value).slice(0, 20) + "..."
                                    : String(value)}
                                </span>
                              </div>
                            ))}
                        </div>
                      )}
                    </div>

                    <div className="flex flex-col items-end gap-1">
                      {agent.status === "completed" ? (
                        <CheckCircle className="w-4 h-4 text-green-400" />
                      ) : (
                        <Clock className="w-4 h-4 text-gray-400" />
                      )}
                      {agent.processingTime && (
                        <span className="text-xs text-gray-500 bg-white/10 px-2 py-1 rounded-full">
                          {agent.processingTime}ms
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};

const MessageSources = ({ sources }) => {
  if (!sources || sources.length === 0) return null;

  return (
    <div className="mt-4 pt-4 border-t border-white/10">
      <div className="flex items-center gap-2 mb-3">
        <div className="w-5 h-5 bg-gradient-to-r from-green-500 to-emerald-500 rounded-full flex items-center justify-center">
          <ExternalLink className="w-3 h-3 text-white" />
        </div>
        <span className="text-sm font-medium text-green-400">
          Verified Sources
        </span>
        <span className="text-xs text-gray-500">
          ({sources.length} articles)
        </span>
      </div>

      <div className="space-y-2">
        {sources.map((source, index) => (
          <a
            key={index}
            href={source.url}
            target="_blank"
            rel="noopener noreferrer"
            className="group flex items-center gap-3 p-3 bg-white/5 hover:bg-white/10 rounded-xl border border-white/10 transition-all duration-200 hover:scale-[1.02]"
          >
            <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-lg flex items-center justify-center flex-shrink-0">
              <Globe className="w-4 h-4 text-white" />
            </div>

            <div className="flex-1 min-w-0">
              <h4 className="text-sm font-medium text-white group-hover:text-cyan-300 transition-colors truncate">
                {source.title}
              </h4>
              <div className="flex items-center gap-2 mt-1">
                <span className="text-xs text-gray-400">
                  {source.source_name}
                </span>
                {source.published_at && (
                  <>
                    <span className="text-xs text-gray-500">•</span>
                    <span className="text-xs text-gray-500">
                      {new Date(source.published_at).toLocaleDateString()}
                    </span>
                  </>
                )}
              </div>
            </div>

            {source.relevance_score && (
              <div className="flex items-center gap-1 bg-green-900/30 px-2 py-1 rounded-full">
                <Star className="w-3 h-3 text-green-400" />
                <span className="text-xs text-green-400 font-medium">
                  {Math.round(source.relevance_score * 100)}%
                </span>
              </div>
            )}

            <ExternalLink className="w-4 h-4 text-gray-400 group-hover:text-cyan-300 transition-colors flex-shrink-0" />
          </a>
        ))}
      </div>
    </div>
  );
};

// Main Chat Component
const Chat = () => {
  // State Management
  const [user] = useAtom(userAtom);
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [pendingAnalysis, setPendingAnalysis] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState(null);
  const [isLoadingHistory, setIsLoadingHistory] = useState(true);
  const [showSettings, setShowSettings] = useState(false);
  const [reactions, setReactions] = useState({});
  const [bookmarks, setBookmarks] = useState(new Set());

  // Refs
  const navigate = useNavigate();
  const messagesEndRef = useRef(null);
  const textAreaRef = useRef(null);
  const eventSourceRef = useRef(null);
  const messagesContainerRef = useRef(null);

  // Custom Hooks
  useAutosizeTextArea(textAreaRef, inputValue);
  const { toasts, addToast, removeToast } = useToast();
  const { enabled: soundEnabled, playSound, toggleSound } = useSound();

  // Auto-scroll with user control
  const scrollToBottom = useCallback((smooth = true) => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({
        behavior: smooth ? "smooth" : "auto",
        block: "end"
      });
    }
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, pendingAnalysis, scrollToBottom]);

  // Initialize chat
  useEffect(() => {
    const init = async () => {
      try {
        await loadChatHistory();
        setupSSEConnection();
      } catch (error) {
        addToast("Failed to initialize chat", "error");
      }
    };
    init();
    return cleanup;
  }, []);

  // Load chat history with error handling
  const loadChatHistory = async () => {
    setIsLoadingHistory(true);
    try {
      const data = await ChatApi.loadChatHistory(50);
      if (data?.messages?.length > 0) {
        const formattedMessages = data.messages.map((msg) => ({
          ...msg,
          id: msg.id || Date.now().toString(),
          type: msg.type || (msg.role === "user" ? "user" : "bot"),
          timestamp: msg.timestamp || new Date().toISOString(),
          content: msg.content || ""
        }));
        setMessages(formattedMessages);
        addToast(
          `Loaded ${formattedMessages.length} previous messages`,
          "success"
        );
      } else {
        showWelcomeMessage();
      }
    } catch (err) {
      console.error("Error loading chat history:", err);
      addToast("Failed to load chat history", "error");
      showWelcomeMessage();
    } finally {
      setIsLoadingHistory(false);
    }
  };

  // Enhanced SSE connection with retry logic
  const setupSSEConnection = useCallback(() => {
    const token = localStorage.getItem("Infiya_token");
    if (!token) {
      addToast("Authentication token missing", "error");
      navigate("/");
      return;
    }

    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }

    try {
      const sseUrl = `http://localhost:8000/api/chats/stream?token=${token}`;
      const eventSource = new EventSource(sseUrl);

      eventSource.onopen = () => {
        setIsConnected(true);
        setError(null);
        addToast("Connected to Infiya", "success", 2000);
      };

      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          handleSSEMessage(data);
        } catch (parseError) {
          console.error("Failed to parse SSE message:", parseError);
          addToast("Received invalid message from server", "error");
        }
      };

      eventSource.onerror = (error) => {
        console.error("SSE connection error:", error);
        setIsConnected(false);
        eventSource.close();

        // Retry connection after delay
        setTimeout(() => {
          if (
            !eventSourceRef.current ||
            eventSourceRef.current.readyState === EventSource.CLOSED
          ) {
            setupSSEConnection();
          }
        }, 5000);
      };

      eventSourceRef.current = eventSource;
    } catch (error) {
      console.error("Failed to establish SSE connection:", error);
      setError("Failed to connect to Infiya's servers");
      addToast("Connection failed. Retrying...", "error");
    }
  }, [navigate, addToast]);

  // Enhanced SSE message handling
  const handleSSEMessage = useCallback(
    (data) => {
      console.log("SSE Message received:", data);

      switch (data.type) {
        case "connection_established":
          setIsConnected(true);
          setError(null);
          break;

        case "workflow_started":
          setPendingAnalysis({ agents: [], workflowId: data.workflow_id });
          break;

        case "agent_update":
          setPendingAnalysis((prev) => {
            if (!prev)
              return {
                agents: [
                  {
                    agent: data.agent_name,
                    status: data.status,
                    message: data.message,
                    progress: data.progress
                  }
                ]
              };

            const agents = [...prev.agents];
            const existingIndex = agents.findIndex(
              (a) => a.agent === data.agent_name
            );

            if (existingIndex !== -1) {
              agents[existingIndex] = {
                agent: data.agent_name,
                status: data.status,
                message: data.message,
                progress: data.progress
              };
            } else {
              agents.push({
                agent: data.agent_name,
                status: data.status,
                message: data.message,
                progress: data.progress
              });
            }

            return { ...prev, agents };
          });
          break;

        case "assistant_response":
          const newMessage = {
            ...data.message,
            id: data.message.id || Date.now().toString(),
            type: "bot",
            timestamp: data.message.timestamp || new Date().toISOString(),
            agentAnalysis: pendingAnalysis?.agents || []
          };

          setMessages((prev) => [...prev, newMessage]);
          setIsSending(false);
          setPendingAnalysis(null);
          playSound("receive");
          break;

        case "workflow_completed":
          // Create assistant message with final response
          if (data.final_response) {
            const assistantMessage = {
              id: Date.now().toString(),
              type: "bot",
              content: data.final_response,
              timestamp: new Date().toISOString(),
              agentAnalysis: pendingAnalysis?.agents || [],
              workflow_stats: data.workflow_stats || null
            };
            setMessages((prev) => [...prev, assistantMessage]);
            playSound("receive");
          }

          setIsSending(false);
          setPendingAnalysis(null);
          addToast("Analysis complete!", "success", 2000);
          break;

        case "workflow_error":
          const errorMessage =
            data.error || "An error occurred during analysis";

          // Create error message as chat response
          const errorChatMessage = {
            id: Date.now().toString(),
            type: "bot",
            content: `I apologize, but I encountered an issue while processing your request: **${errorMessage}**\n\nPlease try again in a few moments. If the problem persists, it might be due to high demand on our AI services.`,
            timestamp: new Date().toISOString(),
            agentAnalysis: pendingAnalysis?.agents || []
          };

          setMessages((prev) => [...prev, errorChatMessage]);
          setIsSending(false);
          setPendingAnalysis(null);
          playSound("receive");
          break;

        case "heartbeat":
          setIsConnected(true);
          break;

        default:
          console.log("Unknown SSE message type:", data.type);
      }
    },
    [pendingAnalysis, addToast, playSound]
  );

  // Enhanced message sending
  const handleSendMessage = async () => {
    const content = inputValue.trim();
    if (!content || isSending) return;

    if (!isConnected) {
      addToast(
        "Not connected to Infiya. Please wait for reconnection.",
        "error"
      );
      return;
    }

    setIsSending(true);
    setInputValue("");
    setError(null);

    const userMessage = {
      id: Date.now().toString(),
      type: "user",
      content,
      timestamp: new Date().toISOString()
    };

    setMessages((prev) => [...prev, userMessage]);
    playSound("send");

    try {
      await ChatApi.sendMessage(content);
      addToast("Message sent to Infiya", "success", 1500);
    } catch (err) {
      console.error("Failed to send message:", err);
      const errorMessage = err.message || "Failed to send message";
      setError(errorMessage);
      addToast(errorMessage, "error");

      // Remove the user message if sending failed
      setMessages((prev) => prev.filter((m) => m.id !== userMessage.id));
      setIsSending(false);
      playSound("error");
    }
  };

  // Message actions
  const handleMessageReaction = useCallback(
    (messageId, reactionType) => {
      setReactions((prev) => ({
        ...prev,
        [messageId]: {
          ...prev[messageId],
          [reactionType]: (prev[messageId]?.[reactionType] || 0) + 1
        }
      }));
      addToast(
        `${reactionType === "like" ? "Liked" : "Disliked"} message`,
        "success",
        1000
      );
    },
    [addToast]
  );

  const handleBookmark = useCallback(
    (message) => {
      const newBookmarks = new Set(bookmarks);
      if (newBookmarks.has(message.id)) {
        newBookmarks.delete(message.id);
        addToast("Removed bookmark", "success", 1000);
      } else {
        newBookmarks.add(message.id);
        addToast("Message bookmarked", "success", 1000);
      }
      setBookmarks(newBookmarks);
    },
    [bookmarks, addToast]
  );

  const handleShare = useCallback(
    async (message) => {
      try {
        if (navigator.share) {
          await navigator.share({
            title: "Infiya AI Response",
            text: message.content,
            url: window.location.href
          });
        } else {
          await navigator.clipboard.writeText(message.content);
          addToast("Message copied to clipboard", "success");
        }
      } catch (error) {
        console.error("Failed to share:", error);
        addToast("Failed to share message", "error");
      }
    },
    [addToast]
  );

  const cleanup = () => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
  };

  const handleLogout = () => {
    cleanup();
    AuthApi.logout();
    navigate("/");
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const showWelcomeMessage = () => {
    const welcomeMessage = {
      id: "welcome",
      type: "bot",
      content: `# Hiii I'm Infiya !! 👋

I'm your personalized AI news anchor, powered by advanced multi-agent analysis technology.
*Ready to get started? Ask me anything!* ✨`,
      timestamp: new Date().toISOString(),
      sources: []
    };
    setMessages([welcomeMessage]);
  };

  const clearChat = async () => {
    try {
      await ChatApi.clearChat();
      setMessages([]);
      showWelcomeMessage();
      addToast("Chat history cleared", "success");
    } catch (error) {
      addToast("Failed to clear chat", "error");
    }
  };

  if (isLoadingHistory) {
    return (
      <div className="h-screen w-screen bg-slate-900 flex items-center justify-center">
        <div className="text-center space-y-6">
          <div className="relative">
            <div className="w-20 h-20 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center mx-auto">
              <Bot className="w-10 h-10 text-white" />
            </div>
            <div className="absolute inset-0 w-20 h-20 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full mx-auto animate-ping opacity-20"></div>
          </div>
          <div>
            <h2 className="text-2xl font-bold text-white mb-2">
              Initializing Infiya
            </h2>
            <p className="text-purple-300 mb-4">
              Setting up your personalized AI news experience...
            </p>
            <div className="w-12 h-12 border-4 border-purple-500 border-t-transparent rounded-full animate-spin mx-auto"></div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      className={`h-screen w-screen bg-slate-900 flex flex-col font-sans transition-all duration-300 `}
    >
      {/* Toast Container */}
      <ToastContainer toasts={toasts} onRemove={removeToast} />

      {/* Background */}
      <div className="absolute inset-0 overflow-hidden z-0">
        <div className="absolute top-0 -left-4 w-96 h-96 bg-purple-600/20 rounded-full mix-blend-multiply filter blur-3xl opacity-70 animate-blob"></div>
        <div className="absolute -bottom-8 right-20 w-96 h-96 bg-blue-600/20 rounded-full mix-blend-multiply filter blur-3xl opacity-70 animate-blob animation-delay-4000"></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-pink-600/10 rounded-full mix-blend-multiply filter blur-3xl opacity-50 animate-blob animation-delay-2000"></div>
      </div>

      {/* Enhanced Header */}
      <header className="relative z-10 bg-slate-900/80 backdrop-blur-xl border-b border-white/10 p-4 shadow-2xl">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="relative group">
              <div className="w-12 h-12 bg-gradient-to-r from-purple-500 via-pink-500 to-purple-600 rounded-full flex items-center justify-center shadow-lg group-hover:shadow-purple-500/25 transition-all duration-300">
                <Bot className="w-7 h-7 text-white" />
              </div>
              <div
                className={`absolute -bottom-1 -right-1 w-4 h-4 rounded-full border-2 border-slate-900 transition-all duration-300 ${
                  isConnected
                    ? "bg-green-400 shadow-lg shadow-green-400/50"
                    : "bg-red-400 animate-pulse"
                }`}
              />
              {isConnected && (
                <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-green-400 rounded-full animate-ping opacity-20"></div>
              )}
            </div>

            <div>
              <h1 className="text-xl font-bold text-white tracking-tight">
                Infiya
              </h1>
              <div className="flex items-center gap-2">
                <p className="text-sm text-purple-300 font-medium">
                  {user?.preferences?.news_personality
                    ?.replace(/_/g, " ")
                    .replace(/\b\w/g, (l) => l.toUpperCase()) ||
                    "AI News Assistant"}
                </p>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <ConnectionStatus
              isConnected={isConnected}
              onRetry={setupSSEConnection}
            />

            {/* Sound Toggle */}
            <button
              onClick={toggleSound}
              className="p-2 text-gray-300 hover:text-white hover:bg-white/10 rounded-lg transition-colors"
              title={soundEnabled ? "Disable sounds" : "Enable sounds"}
            >
              {soundEnabled ? (
                <Volume2 className="w-5 h-5" />
              ) : (
                <VolumeX className="w-5 h-5" />
              )}
            </button>

            {/* Clear Chat */}
            <button
              onClick={clearChat}
              className="p-2 text-gray-300 hover:text-white hover:bg-white/10 rounded-lg transition-colors"
              title="Clear Chat"
            >
              <Trash2 className="w-5 h-5" />
            </button>

            <button
              onClick={() => setShowSettings(true)}
              className="p-2 text-gray-300 hover:text-white hover:bg-white/10 rounded-lg transition-colors"
              title="Settings"
            >
              <Settings className="w-5 h-5" />
            </button>

            <button
              onClick={handleLogout}
              className="p-2 text-gray-300 hover:text-white hover:bg-white/10 rounded-lg transition-colors"
              title="Logout"
            >
              <LogOut className="w-5 h-5" />
            </button>
          </div>
        </div>
      </header>

      {/* Error Banner */}
      {error && (
        <div className="relative z-10 bg-gradient-to-r from-red-600/90 to-red-500/90 backdrop-blur-sm text-white p-4 animate-slide-down">
          <div className="max-w-6xl mx-auto flex items-center justify-between">
            <div className="flex items-center gap-3">
              <AlertCircle className="w-5 h-5 flex-shrink-0" />
              <span className="font-medium">{error}</span>
            </div>
            <button
              onClick={() => setError(null)}
              className="text-white hover:text-red-200 transition-colors p-1"
            >
              <div className="w-6 h-6 flex items-center justify-center font-bold">
                ×
              </div>
            </button>
          </div>
        </div>
      )}

      {/* Enhanced Chat Messages */}
      <main
        ref={messagesContainerRef}
        className="flex-1 overflow-y-auto p-6 custom-scrollbar"
      >
        <div className="max-w-6xl mx-auto space-y-8">
          {messages.map((msg, index) => {
            const isUser = msg.type === "user";
            const parsedContent = DOMPurify.sanitize(
              marked.parse(msg.content || "")
            );

            return (
              <div
                key={msg.id}
                className={`group flex items-start gap-4 animate-fade-in-up opacity-0 ${
                  isUser ? "justify-end" : ""
                }`}
                style={{
                  animationDelay: `${index * 50}ms`,
                  animationFillMode: "forwards"
                }}
              >
                {!isUser && (
                  <div className="flex-shrink-0">
                    <div className="w-10 h-10 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center shadow-lg hover:shadow-purple-500/25 transition-all duration-300">
                      <Bot className="w-5 h-5 text-white" />
                    </div>
                  </div>
                )}

                <div
                  className={`flex flex-col max-w-4xl ${
                    isUser ? "items-end" : "items-start"
                  }`}
                >
                  <div
                    className={`relative p-6 rounded-3xl backdrop-blur-lg border transition-all duration-300 hover:scale-[1.01] ${
                      isUser
                        ? "bg-gradient-to-br from-purple-600/90 to-pink-600/90 text-white border-white/20 rounded-br-lg shadow-xl shadow-purple-500/10"
                        : "bg-slate-800/80 text-gray-100 border-white/10 rounded-bl-lg shadow-xl shadow-slate-900/10"
                    }`}
                  >
                    {/* Message Content */}
                    <div
                      className="prose prose-sm prose-invert max-w-none leading-relaxed"
                      dangerouslySetInnerHTML={{ __html: parsedContent }}
                    />

                    {!isUser && msg.workflow_stats && (
                      <WorkflowStats stats={msg.workflow_stats} />
                    )}

                    {/* Analysis */}
                    {!isUser && msg.agentAnalysis && (
                      <EnhancedMessageAnalysis
                        analysis={msg.agentAnalysis}
                        messageId={msg.id}
                      />
                    )}

                    {/* Message Actions */}
                    <div className="absolute top-3 right-3 opacity-0 group-hover:opacity-100 transition-opacity">
                      <MessageActions
                        message={msg}
                        onCopy={() =>
                          addToast("Copied to clipboard", "success", 1000)
                        }
                        onShare={handleShare}
                        onBookmark={handleBookmark}
                        onReact={handleMessageReaction}
                      />
                    </div>
                  </div>

                  {/* Message Footer */}
                  <div className="flex items-center gap-3 mt-2 px-2">
                    <span className="text-xs text-gray-500 font-medium">
                      {new Date(msg.timestamp).toLocaleTimeString([], {
                        hour: "2-digit",
                        minute: "2-digit"
                      })}
                    </span>

                    {reactions[msg.id] && (
                      <div className="flex items-center gap-2 text-xs">
                        {reactions[msg.id].like > 0 && (
                          <div className="flex items-center gap-1 bg-green-900/30 px-2 py-1 rounded-full">
                            <ThumbsUp className="w-3 h-3 text-green-400" />
                            <span className="text-green-400">
                              {reactions[msg.id].like}
                            </span>
                          </div>
                        )}
                        {reactions[msg.id].dislike > 0 && (
                          <div className="flex items-center gap-1 bg-red-900/30 px-2 py-1 rounded-full">
                            <ThumbsDown className="w-3 h-3 text-red-400" />
                            <span className="text-red-400">
                              {reactions[msg.id].dislike}
                            </span>
                          </div>
                        )}
                      </div>
                    )}

                    {bookmarks.has(msg.id) && (
                      <Bookmark className="w-4 h-4 text-yellow-400 fill-current" />
                    )}
                  </div>
                </div>

                {isUser && (
                  <div className="flex-shrink-0">
                    <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-full flex items-center justify-center shadow-lg hover:shadow-blue-500/25 transition-all duration-300">
                      <User className="w-5 h-5 text-white" />
                    </div>
                  </div>
                )}
              </div>
            );
          })}

          {/* Enhanced Thinking Indicator */}
          {isSending && (
            <EnhancedThinkingIndicator
              agentProgress={pendingAnalysis?.agents || []}
            />
          )}

          <div ref={messagesEndRef} />
        </div>
      </main>

      {/* Enhanced Input Area */}
      <footer className="relative z-10 bg-slate-900/80 backdrop-blur-xl border-t border-white/10 p-6">
        <div className="max-w-6xl mx-auto">
          <div className="relative flex items-end gap-4 bg-slate-800/80 backdrop-blur-lg rounded-3xl border border-white/10 p-3 shadow-2xl focus-within:border-purple-500/50 transition-all duration-300">
            {/* Input Field */}
            <div className="flex-1 px-3">
              <textarea
                ref={textAreaRef}
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask Infiya anything about current events..."
                className="w-full bg-transparent text-white placeholder-gray-400 resize-none focus:outline-none max-h-40 custom-scrollbar text-base leading-relaxed py-3"
                rows="1"
                disabled={isSending || !isConnected}
              />
            </div>

            {/* Action Buttons */}
            <div className="flex items-center gap-2">
              <button
                className="p-3 text-gray-400 hover:text-white transition-all duration-200 rounded-xl hover:bg-white/10 hover:scale-105"
                disabled={true}
                title="Voice input (coming soon)"
              >
                <Mic className="w-5 h-5" />
              </button>

              <button
                onClick={handleSendMessage}
                disabled={!inputValue.trim() || isSending || !isConnected}
                className="group relative p-4 rounded-2xl transition-all duration-300 flex items-center justify-center
                           disabled:bg-gray-600 disabled:text-gray-400 disabled:cursor-not-allowed disabled:scale-100
                           enabled:bg-gradient-to-r enabled:from-purple-500 enabled:to-pink-500 enabled:text-white 
                           enabled:hover:scale-110 enabled:hover:shadow-lg enabled:hover:shadow-purple-500/25
                           enabled:active:scale-95"
              >
                {isSending ? (
                  <div className="w-6 h-6 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                ) : (
                  <>
                    <Send className="w-6 h-6 group-hover:translate-x-0.5 transition-transform duration-200" />
                    <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-purple-400 to-pink-400 opacity-0 group-hover:opacity-20 transition-opacity duration-300"></div>
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Footer Info */}
          <div className="flex items-center justify-between mt-4">
            <p className="text-xs text-gray-500">
              ✨ Infiya uses advanced AI agents to analyze and summarize news
              for you
            </p>

            <div className="flex items-center gap-4 text-xs text-gray-500">
              <div className="flex items-center gap-2">
                <MessageCircle className="w-3 h-3" />
                <span>{messages.length} messages</span>
              </div>

              {bookmarks.size > 0 && (
                <div className="flex items-center gap-2">
                  <Bookmark className="w-3 h-3" />
                  <span>{bookmarks.size} bookmarked</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </footer>

      {/* Settings Dialog */}
      <SettingsDialog
        isOpen={showSettings}
        onClose={() => setShowSettings(false)}
      />
    </div>
  );
};

export default Chat;
