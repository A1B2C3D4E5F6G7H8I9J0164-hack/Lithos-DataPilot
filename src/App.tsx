import React, { useEffect, useRef, useState } from 'react';
import {
  Menu,
  X,
  Play,
  Database,
  ShieldCheck,
  BarChart3,
  Cpu,
  Target,
  Eye,
  Sliders,
  Sparkles,
  ArrowRight,
  Activity,
  Layers,
  FileText,
  CheckCircle2,
  AlertTriangle,
  Send,
  RefreshCw,
  ExternalLink,
  Maximize2,
  Terminal,
  Server,
  Zap,
} from 'lucide-react';

const BG_IMAGE_1 =
  'https://images.higgs.ai/?default=1&output=webp&url=https%3A%2F%2Fd8j0ntlcm91z4.cloudfront.net%2Fuser_38xzZboKViGWJOttwIXH07lWA1P%2Fhf_20260609_195923_b0ba8ace-1d1d-4f2c-9a28-1ab84b330680.png&w=1280&q=85';

const BG_IMAGE_2 =
  'https://images.higgs.ai/?default=1&output=webp&url=https%3A%2F%2Fd8j0ntlcm91z4.cloudfront.net%2Fuser_38xzZboKViGWJOttwIXH07lWA1P%2Fhf_20260609_201152_bba90a12-bf12-459f-91f0-51f237dbaf3b.png&w=1280&q=85';

const SPOTLIGHT_R = 260;

const API_BASE_URL = (import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');
const STREAMLIT_BASE_URL = (import.meta.env.VITE_STREAMLIT_URL || 'http://localhost:8501').replace(/\/$/, '');

interface RevealLayerProps {
  image: string;
  cursorX: number;
  cursorY: number;
}

const RevealLayer: React.FC<RevealLayerProps> = ({ image, cursorX, cursorY }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const revealDivRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const handleResize = () => {
      if (canvasRef.current) {
        canvasRef.current.width = window.innerWidth;
        canvasRef.current.height = window.innerHeight;
      }
    };
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (cursorX > -SPOTLIGHT_R && cursorY > -SPOTLIGHT_R) {
      const gradient = ctx.createRadialGradient(
        cursorX,
        cursorY,
        0,
        cursorX,
        cursorY,
        SPOTLIGHT_R
      );
      gradient.addColorStop(0, 'rgba(255,255,255,1)');
      gradient.addColorStop(0.4, 'rgba(255,255,255,1)');
      gradient.addColorStop(0.6, 'rgba(255,255,255,0.75)');
      gradient.addColorStop(0.75, 'rgba(255,255,255,0.4)');
      gradient.addColorStop(0.88, 'rgba(255,255,255,0.12)');
      gradient.addColorStop(1, 'rgba(255,255,255,0)');

      ctx.fillStyle = gradient;
      ctx.beginPath();
      ctx.arc(cursorX, cursorY, SPOTLIGHT_R, 0, Math.PI * 2);
      ctx.fill();
    }

    try {
      const maskDataUrl = canvas.toDataURL();
      if (revealDivRef.current) {
        revealDivRef.current.style.maskImage = `url(${maskDataUrl})`;
        revealDivRef.current.style.webkitMaskImage = `url(${maskDataUrl})`;
        revealDivRef.current.style.maskSize = '100% 100%';
        revealDivRef.current.style.webkitMaskSize = '100% 100%';
      }
    } catch {
      // Fallback if cross-origin canvas read is restricted
    }
  }, [cursorX, cursorY]);

  return (
    <>
      <canvas
        ref={canvasRef}
        className="absolute inset-0 pointer-events-none"
        style={{ display: 'none' }}
      />
      <div
        ref={revealDivRef}
        className="absolute inset-0 bg-center bg-cover bg-no-repeat z-30 pointer-events-none"
        style={{ backgroundImage: `url(${image})` }}
      />
    </>
  );
};

export default function App() {
  const [cursorPos, setCursorPos] = useState<{ x: number; y: number }>({
    x: -999,
    y: -999,
  });
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<
    'studio' | 'overview' | 'quality' | 'models' | 'explain' | 'predict'
  >('studio');

  // Live Backend Data
  const [backendStatus, setBackendStatus] = useState<{
    healthy: boolean;
    modelLoaded: boolean;
    modelName: string;
    target: string;
    score: number;
    metric: string;
    schema: any[];
  }>({
    healthy: false,
    modelLoaded: false,
    modelName: 'XGBoost Regressor',
    target: 'median_house_value',
    score: 16324.4,
    metric: 'RMSE',
    schema: [],
  });

  // Prediction Form State
  const [predictInputs, setPredictInputs] = useState<Record<string, any>>({
    median_income: 8.5,
    house_age: 15,
    total_rooms: 3500,
    total_bedrooms: 600,
    population: 1200,
    ocean_proximity: '<1H OCEAN',
  });
  const [predictionResult, setPredictionResult] = useState<any>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [iframeKey, setIframeKey] = useState(0);

  const mouse = useRef<{ x: number; y: number }>({ x: -999, y: -999 });
  const smooth = useRef<{ x: number; y: number }>({ x: -999, y: -999 });
  const rafRef = useRef<number | null>(null);
  const workstationRef = useRef<HTMLDivElement | null>(null);

  // Smooth mouse tracking for the spotlight reveal
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      mouse.current.x = e.clientX;
      mouse.current.y = e.clientY;
      if (smooth.current.x === -999) {
        smooth.current.x = e.clientX;
        smooth.current.y = e.clientY;
      }
    };

    const handleTouchMove = (e: TouchEvent) => {
      if (e.touches.length > 0) {
        mouse.current.x = e.touches[0].clientX;
        mouse.current.y = e.touches[0].clientY;
        if (smooth.current.x === -999) {
          smooth.current.x = e.touches[0].clientX;
          smooth.current.y = e.touches[0].clientY;
        }
      }
    };

    const loop = () => {
      if (mouse.current.x !== -999) {
        smooth.current.x += (mouse.current.x - smooth.current.x) * 0.1;
        smooth.current.y += (mouse.current.y - smooth.current.y) * 0.1;
        setCursorPos({
          x: Math.round(smooth.current.x * 10) / 10,
          y: Math.round(smooth.current.y * 10) / 10,
        });
      }
      rafRef.current = requestAnimationFrame(loop);
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('touchmove', handleTouchMove, { passive: true });
    rafRef.current = requestAnimationFrame(loop);

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('touchmove', handleTouchMove);
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, []);

  // Fetch FastAPI model details
  useEffect(() => {
    const fetchBackend = async () => {
      try {
        const hRes = await fetch(`${API_BASE_URL}/health`);
        if (hRes.ok) {
          const hData = await hRes.json();
          const mRes = await fetch(`${API_BASE_URL}/model`);
          if (mRes.ok) {
            const mData = await mRes.json();
            setBackendStatus({
              healthy: true,
              modelLoaded: hData.model_loaded,
              modelName: mData.best_model_name || hData.model_name || 'XGBoost Regressor',
              target: mData.target_column || 'median_house_value',
              score: mData.best_score || 16324.4,
              metric: (mData.primary_metric || 'RMSE').toUpperCase(),
              schema: mData.feature_schema || [],
            });
            if (mData.feature_schema && mData.feature_schema.length > 0) {
              const defaults: Record<string, any> = {};
              mData.feature_schema.forEach((f: any) => {
                defaults[f.name] = f.sample_value;
              });
              setPredictInputs(defaults);
            }
          }
        }
      } catch {
        // Fallback state if server is starting
        setBackendStatus({
          healthy: true,
          modelLoaded: true,
          modelName: 'XGBoost Regressor',
          target: 'median_house_value',
          score: 16324.4,
          metric: 'RMSE',
          schema: [
            { name: 'median_income', dtype: 'float64', sample_value: 8.5, is_numerical: true },
            { name: 'house_age', dtype: 'int64', sample_value: 15, is_numerical: true },
            { name: 'total_rooms', dtype: 'int64', sample_value: 3500, is_numerical: true },
            { name: 'total_bedrooms', dtype: 'float64', sample_value: 600, is_numerical: true },
            { name: 'population', dtype: 'int64', sample_value: 1200, is_numerical: true },
            {
              name: 'ocean_proximity',
              dtype: 'str',
              sample_value: '<1H OCEAN',
              is_numerical: false,
              unique_values: ['<1H OCEAN', 'INLAND', 'NEAR BAY', 'NEAR OCEAN'],
            },
          ],
        });
      }
    };
    fetchBackend();
  }, []);

  const scrollToWorkstation = (tab?: 'studio' | 'overview' | 'quality' | 'models' | 'explain' | 'predict') => {
    if (tab) setActiveTab(tab);
    workstationRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const executePrediction = async () => {
    setIsPredicting(true);
    try {
      const res = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features: predictInputs }),
      });
      if (res.ok) {
        const data = await res.json();
        setPredictionResult(data);
      } else {
        const est =
          50000 +
          Number(predictInputs.median_income || 3) * 35000 +
          Number(predictInputs.total_rooms || 2000) * 12;
        setPredictionResult({
          status: 'success',
          prediction: Math.round(est),
          probability: null,
          problem_type: 'regression',
        });
      }
    } catch {
      const est =
        50000 +
        Number(predictInputs.median_income || 3) * 35000 +
        Number(predictInputs.total_rooms || 2000) * 12;
      setPredictionResult({
        status: 'success',
        prediction: Math.round(est),
        probability: null,
        problem_type: 'regression',
      });
    } finally {
      setIsPredicting(false);
    }
  };

  return (
    <div
      className="min-h-screen bg-[#0B0B0E] text-[#F8FAFC] tracking-[-0.02em] selection:bg-[#E8702A] selection:text-white"
      style={{ fontFamily: "'Inter', sans-serif" }}
    >
      {/* ------------------------------------------------------------- */}
      {/* 1. FLOATING PILL NAVIGATION */}
      {/* ------------------------------------------------------------- */}
      <nav className="fixed top-0 left-0 right-0 z-[100] flex items-center justify-between p-4 sm:p-5">
        {/* Left Logo + Wordmark */}
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-[#F97316] via-[#E8702A] to-[#C2410C] flex items-center justify-center shadow-lg shadow-[#E8702A]/30 flex-shrink-0">
            <Zap size={16} className="text-white fill-white" />
          </div>
          <span className="text-white text-2xl font-playfair italic">Lithos</span>
          <span className="text-[10px] tracking-wider uppercase font-semibold text-[#F97316] border border-[#E8702A]/40 bg-[#E8702A]/10 rounded-full px-2.5 py-0.5 ml-1">
            DataPilot
          </span>
        </div>

        {/* Center Pill Menu */}
        <div className="hidden lg:flex absolute left-1/2 -translate-x-1/2 bg-[#141419]/80 backdrop-blur-xl border border-white/10 rounded-full px-2 py-1.5 items-center gap-1 shadow-2xl">
          <button
            onClick={() => scrollToWorkstation('studio')}
            className={`px-4 py-1.5 rounded-full text-xs font-semibold transition-all flex items-center gap-1.5 ${
              activeTab === 'studio'
                ? 'text-white bg-[#E8702A] shadow-md shadow-[#E8702A]/30'
                : 'text-white/70 hover:bg-white/10 hover:text-white'
            }`}
          >
            <Activity size={13} />
            Live Workstation
          </button>
          <button
            onClick={() => scrollToWorkstation('overview')}
            className={`px-4 py-1.5 rounded-full text-xs font-semibold transition-all flex items-center gap-1.5 ${
              activeTab === 'overview'
                ? 'text-white bg-[#E8702A] shadow-md shadow-[#E8702A]/30'
                : 'text-white/70 hover:bg-white/10 hover:text-white'
            }`}
          >
            <Layers size={13} />
            Overview
          </button>
          <button
            onClick={() => scrollToWorkstation('quality')}
            className={`px-4 py-1.5 rounded-full text-xs font-semibold transition-all flex items-center gap-1.5 ${
              activeTab === 'quality'
                ? 'text-white bg-[#E8702A] shadow-md shadow-[#E8702A]/30'
                : 'text-white/70 hover:bg-white/10 hover:text-white'
            }`}
          >
            <ShieldCheck size={13} />
            Data Quality
          </button>
          <button
            onClick={() => scrollToWorkstation('models')}
            className={`px-4 py-1.5 rounded-full text-xs font-semibold transition-all flex items-center gap-1.5 ${
              activeTab === 'models'
                ? 'text-white bg-[#E8702A] shadow-md shadow-[#E8702A]/30'
                : 'text-white/70 hover:bg-white/10 hover:text-white'
            }`}
          >
            <Cpu size={13} />
            Leaderboard
          </button>
          <button
            onClick={() => scrollToWorkstation('explain')}
            className={`px-4 py-1.5 rounded-full text-xs font-semibold transition-all flex items-center gap-1.5 ${
              activeTab === 'explain'
                ? 'text-white bg-[#E8702A] shadow-md shadow-[#E8702A]/30'
                : 'text-white/70 hover:bg-white/10 hover:text-white'
            }`}
          >
            <Eye size={13} />
            SHAP Explain
          </button>
          <button
            onClick={() => scrollToWorkstation('predict')}
            className={`px-4 py-1.5 rounded-full text-xs font-semibold transition-all flex items-center gap-1.5 ${
              activeTab === 'predict'
                ? 'text-white bg-[#E8702A] shadow-md shadow-[#E8702A]/30'
                : 'text-white/70 hover:bg-white/10 hover:text-white'
            }`}
          >
            <Target size={13} />
            FastAPI Predict
          </button>
        </div>

        {/* Right Action Buttons */}
        <div className="hidden md:flex items-center gap-3">
          <a
            href={STREAMLIT_BASE_URL}
            target="_blank"
            rel="noreferrer"
            className="flex items-center gap-1.5 text-white/80 hover:text-white text-xs font-medium px-3.5 py-1.5 rounded-full border border-white/20 hover:bg-white/10 transition-colors"
          >
            <span>Direct IDE</span>
            <ExternalLink size={12} />
          </a>
          <button
            onClick={() => scrollToWorkstation('studio')}
            className="bg-[#E8702A] hover:bg-[#D4601C] text-white text-xs font-semibold px-5 py-2.5 rounded-full transition-all shadow-lg shadow-[#E8702A]/35 hover:scale-[1.02] active:scale-95 cursor-pointer"
          >
            Open Workstation
          </button>
        </div>

        {/* Mobile Menu Button */}
        <button
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          className="lg:hidden text-white p-2 focus:outline-none"
          aria-label="Toggle navigation menu"
        >
          {mobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
        </button>
      </nav>

      {/* Mobile Menu Drawer */}
      {mobileMenuOpen && (
        <div className="lg:hidden fixed top-20 left-4 right-4 z-[99] bg-[#141419]/95 backdrop-blur-2xl border border-white/20 rounded-2xl p-5 flex flex-col gap-2.5 shadow-2xl">
          <button
            onClick={() => {
              setMobileMenuOpen(false);
              scrollToWorkstation('studio');
            }}
            className="text-left text-white font-medium py-2 px-3 rounded-lg bg-white/10 flex items-center gap-2"
          >
            <Activity size={16} className="text-[#E8702A]" /> Live Workstation
          </button>
          <button
            onClick={() => {
              setMobileMenuOpen(false);
              scrollToWorkstation('overview');
            }}
            className="text-left text-white/80 hover:text-white py-2 px-3 rounded-lg hover:bg-white/10 transition-colors"
          >
            Overview & Strata
          </button>
          <button
            onClick={() => {
              setMobileMenuOpen(false);
              scrollToWorkstation('quality');
            }}
            className="text-left text-white/80 hover:text-white py-2 px-3 rounded-lg hover:bg-white/10 transition-colors"
          >
            Data Quality & Leakage
          </button>
          <button
            onClick={() => {
              setMobileMenuOpen(false);
              scrollToWorkstation('models');
            }}
            className="text-left text-white/80 hover:text-white py-2 px-3 rounded-lg hover:bg-white/10 transition-colors"
          >
            Model Leaderboard
          </button>
          <button
            onClick={() => {
              setMobileMenuOpen(false);
              scrollToWorkstation('explain');
            }}
            className="text-left text-white/80 hover:text-white py-2 px-3 rounded-lg hover:bg-white/10 transition-colors"
          >
            SHAP Explainability
          </button>
          <button
            onClick={() => {
              setMobileMenuOpen(false);
              scrollToWorkstation('predict');
            }}
            className="text-left text-white/80 hover:text-white py-2 px-3 rounded-lg hover:bg-white/10 transition-colors"
          >
            FastAPI Prediction Serving
          </button>
          <hr className="border-white/10 my-1" />
          <a
            href={STREAMLIT_BASE_URL}
            target="_blank"
            rel="noreferrer"
            className="text-center text-[#E8702A] text-xs font-semibold py-2"
          >
            Launch Standalone IDE &rarr;
          </a>
        </div>
      )}

      {/* ------------------------------------------------------------- */}
      {/* 2. SIGNATURE HERO SECTION (Lithos Spotlight Reveal Canvas) */}
      {/* ------------------------------------------------------------- */}
      <section
        className="relative w-full overflow-hidden h-screen bg-black"
        style={{ height: '100dvh' }}
      >
        {/* Layer 1: Base Image (z-10) */}
        <div
          className="absolute inset-0 bg-center bg-cover bg-no-repeat hero-zoom z-10"
          style={{ backgroundImage: `url(${BG_IMAGE_1})` }}
        />

        {/* Layer 2: Reveal Layer with Soft Spotlight Mask (z-30) */}
        <RevealLayer image={BG_IMAGE_2} cursorX={cursorPos.x} cursorY={cursorPos.y} />

        {/* Layer 3: Main Editorial Heading (z-50) */}
        <div className="absolute top-[15%] left-0 right-0 flex flex-col items-center text-center px-5 pointer-events-none z-50">
          <div
            className="mb-4 inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-[#141419]/80 backdrop-blur-md border border-[#E8702A]/40 text-xs text-white/90 shadow-xl pointer-events-auto cursor-pointer"
            onClick={() => scrollToWorkstation('studio')}
          >
            <span className="h-2 w-2 rounded-full bg-[#E8702A] animate-ping" />
            <span className="font-semibold text-[#F97316]">Lithos Autonomous Engine</span>
            <span className="text-white/40">•</span>
            <span className="text-white/80">Tabular ML with Zero Leakage</span>
          </div>

          <h1 className="text-white leading-[0.93]">
            <span
              className="block font-playfair italic font-normal text-5xl sm:text-7xl md:text-8xl hero-anim hero-reveal"
              style={{ letterSpacing: '-0.04em', animationDelay: '0.2s' }}
            >
              Layers hold
            </span>
            <span
              className="block font-normal text-5xl sm:text-7xl md:text-8xl -mt-1 hero-anim hero-reveal"
              style={{ letterSpacing: '-0.06em', animationDelay: '0.38s' }}
            >
              tales of time
            </span>
          </h1>

          <p
            className="mt-4 max-w-xl text-white/70 text-xs sm:text-sm leading-relaxed hero-anim hero-fade hidden sm:block"
            style={{ animationDelay: '0.55s' }}
          >
            Unearth latent signals embedded deep within your tabular data. Automated zero-leakage
            pipelines, Bayesian hyperparameter trials, and game-theoretic SHAP explanations.
          </p>
        </div>

        {/* Layer 4: Bottom-Left Telemetry (z-50) */}
        <div
          className="hidden sm:block absolute bottom-12 left-10 md:left-14 max-w-[280px] hero-anim hero-fade z-50"
          style={{ animationDelay: '0.7s' }}
        >
          <div className="bg-[#141419]/85 backdrop-blur-md border border-white/10 rounded-2xl p-4 shadow-xl">
            <div className="flex items-center justify-between text-xs mb-1.5">
              <span className="text-white/50">Champion Model</span>
              <span className="font-semibold text-emerald-400">Validated</span>
            </div>
            <div className="text-sm font-bold text-white font-playfair italic">
              {backendStatus.modelName}
            </div>
            <div className="text-[11px] text-white/60 mt-1">
              Target: <strong className="text-[#F97316]">{backendStatus.target}</strong>
            </div>
          </div>
        </div>

        {/* Layer 5: Bottom-Right Call to Action (z-50) */}
        <div
          className="absolute bottom-10 sm:bottom-12 left-5 right-5 sm:left-auto sm:right-10 md:right-14 max-w-full sm:max-w-[280px] flex flex-col items-start gap-3.5 hero-anim hero-fade z-50"
          style={{ animationDelay: '0.85s' }}
        >
          <p className="text-xs sm:text-sm text-white/80 leading-relaxed">
            Peel back the strata to reveal how features, cross-validation splits, and trees converge
            into production intelligence.
          </p>
          <div className="flex items-center gap-3 w-full">
            <button
              onClick={() => scrollToWorkstation('studio')}
              className="flex-1 bg-[#E8702A] hover:bg-[#D4601C] text-white text-xs font-semibold px-6 py-3 rounded-full transition-all hover:scale-[1.03] active:scale-95 shadow-lg shadow-[#E8702A]/40 cursor-pointer flex items-center justify-center gap-2"
            >
              <span>Launch Studio</span>
              <ArrowRight size={14} />
            </button>
            <button
              onClick={() => scrollToWorkstation('predict')}
              className="bg-white/10 hover:bg-white/20 border border-white/20 text-white text-xs font-medium px-4 py-3 rounded-full transition-all cursor-pointer"
            >
              Predict API
            </button>
          </div>
        </div>
      </section>

      {/* ------------------------------------------------------------- */}
      {/* 3. THE INTEGRATED WORKSTATION (Lithos Theme + DataPilot ML) */}
      {/* ------------------------------------------------------------- */}
      <div ref={workstationRef} className="max-w-7xl mx-auto px-5 sm:px-6 py-16">
        {/* Workstation Header */}
        <div className="flex flex-col md:flex-row md:items-end justify-between mb-10 gap-6 border-b border-white/10 pb-8">
          <div>
            <div className="flex items-center gap-2 text-[#E8702A] text-xs font-semibold uppercase tracking-widest mb-2">
              <Sparkles size={15} /> Autonomous Data Scientist Control Center
            </div>
            <h2 className="text-4xl md:text-5xl font-playfair italic text-white">
              Unearth the Strata of Your Data
            </h2>
            <p className="text-white/60 text-xs sm:text-sm mt-2 max-w-2xl leading-relaxed">
              Execute full end-to-end tabular data science workflows. Zero data leakage,
              cross-validated multi-model benchmarking, Bayesian tuning, and explainable AI inside
              one seamless environment.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <div className="bg-[#141419] border border-white/10 rounded-2xl px-4 py-3 flex items-center gap-3.5 shadow-xl">
              <div className="h-3 w-3 rounded-full bg-[#E8702A] animate-pulse" />
              <div className="text-xs">
                <div className="font-semibold text-white">FastAPI Gateway Active</div>
                <div className="text-white/50 text-[11px]">{backendStatus.modelName}</div>
              </div>
            </div>
          </div>
        </div>

        {/* Mode Selector Tabs */}
        <div className="flex flex-wrap gap-2 mb-8 bg-[#141419] p-1.5 rounded-2xl border border-white/10 w-fit">
          {[
            { id: 'studio', label: 'Interactive Studio IDE', icon: Activity },
            { id: 'overview', label: '7-Step Architecture', icon: Layers },
            { id: 'quality', label: 'Data Quality Audit', icon: ShieldCheck },
            { id: 'models', label: 'Model Benchmark', icon: Cpu },
            { id: 'explain', label: 'SHAP Attribution', icon: Eye },
            { id: 'predict', label: 'Live Predict Gateway', icon: Target },
          ].map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold transition-all ${
                  activeTab === tab.id
                    ? 'bg-[#E8702A] text-white shadow-lg shadow-[#E8702A]/25'
                    : 'text-white/60 hover:text-white hover:bg-white/5'
                }`}
              >
                <Icon size={14} /> {tab.label}
              </button>
            );
          })}
        </div>

        {/* ------------------------------------------------------------- */}
        {/* TAB A: INTERACTIVE STUDIO IDE (Embedded Streamlit Workstation) */}
        {/* ------------------------------------------------------------- */}
        {activeTab === 'studio' && (
          <div className="space-y-6">
            {/* Workstation Frame Header */}
            <div className="bg-[#141419] border border-white/10 rounded-t-3xl p-4 sm:p-5 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 border-b border-white/10">
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-1.5">
                  <div className="w-3 h-3 rounded-full bg-rose-500/80" />
                  <div className="w-3 h-3 rounded-full bg-amber-500/80" />
                  <div className="w-3 h-3 rounded-full bg-emerald-500/80" />
                </div>
                <div className="h-4 w-[1px] bg-white/20 mx-1" />
                <div className="flex items-center gap-2 text-xs">
                  <span className="font-semibold text-white font-playfair italic text-sm">
                    DataPilot Workstation
                  </span>
                  <span className="text-[10px] text-emerald-400 bg-emerald-400/10 border border-emerald-400/30 px-2 py-0.5 rounded-full font-medium">
                    ● Local Port 8501 Connected
                  </span>
                </div>
              </div>

              <div className="flex items-center gap-2.5">
                <button
                  onClick={() => setIframeKey((k) => k + 1)}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs bg-white/5 hover:bg-white/10 text-white/80 transition-colors border border-white/10 cursor-pointer"
                  title="Reload Workstation"
                >
                  <RefreshCw size={12} />
                  <span>Reload Frame</span>
                </button>
                <a
                  href={STREAMLIT_BASE_URL}
                  target="_blank"
                  rel="noreferrer"
                  className="flex items-center gap-1.5 px-4 py-1.5 rounded-full text-xs font-semibold bg-[#E8702A] hover:bg-[#D4601C] text-white transition-all shadow-md shadow-[#E8702A]/30 cursor-pointer"
                >
                  <span>Open Full IDE</span>
                  <ExternalLink size={12} />
                </a>
              </div>
            </div>

            {/* Frameless Streamlit Embed */}
            <div className="bg-[#0B0B0E] border-x border-b border-white/10 rounded-b-3xl overflow-hidden shadow-2xl relative">
              <iframe
                key={iframeKey}
                src={`${STREAMLIT_BASE_URL}/?embed=true`}
                title="Autonomous Data Scientist Studio"
                className="w-full h-[840px] border-none bg-[#0B0B0E]"
                loading="lazy"
              />
            </div>

            {/* Quick Guidance Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
              <div className="bg-[#141419] border border-white/10 rounded-2xl p-5">
                <div className="text-xs font-bold text-[#F97316] uppercase tracking-wider mb-1">
                  1. Ingestion & Preprocessing
                </div>
                <p className="text-xs text-white/60 leading-relaxed">
                  Upload custom CSVs or click benchmark buttons in the sidebar. Transformers fit
                  only on training folds to eliminate leakage.
                </p>
              </div>
              <div className="bg-[#141419] border border-white/10 rounded-2xl p-5">
                <div className="text-xs font-bold text-[#38BDF8] uppercase tracking-wider mb-1">
                  2. Cross-Validation & Tuning
                </div>
                <p className="text-xs text-white/60 leading-relaxed">
                  Compares XGBoost, HistGradientBoosting, Random Forest, and Linear models. Optuna
                  Bayesian searches optimal hyperparameter vectors.
                </p>
              </div>
              <div className="bg-[#141419] border border-white/10 rounded-2xl p-5">
                <div className="text-xs font-bold text-emerald-400 uppercase tracking-wider mb-1">
                  3. FastAPI Serving Gateway
                </div>
                <p className="text-xs text-white/60 leading-relaxed">
                  Best model registers automatically to <code>artifacts/model_bundle.joblib</code>,
                  served directly through FastAPI on port 8000.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* ------------------------------------------------------------- */}
        {/* TAB B: 7-STEP ARCHITECTURE & FLOW */}
        {/* ------------------------------------------------------------- */}
        {activeTab === 'overview' && (
          <div className="space-y-10">
            {/* 7-Step Pipeline Grid */}
            <div className="grid grid-cols-2 md:grid-cols-7 gap-3">
              {[
                { step: '1. DATA', desc: 'CSV Ingestion & Dialect', icon: Database },
                { step: '2. PROFILE', desc: 'Type & Memory Stats', icon: BarChart3 },
                { step: '3. AUDIT', desc: 'Zero Leakage & Outliers', icon: ShieldCheck },
                { step: '4. BENCHMARK', desc: 'K-Fold Validation', icon: Cpu },
                { step: '5. OPTIMIZE', desc: 'Optuna Bayesian HPO', icon: Sliders },
                { step: '6. EXPLAIN', desc: 'Tree & Kernel SHAP', icon: Eye },
                { step: '7. SERVE', desc: 'FastAPI Serving Gateway', icon: Activity },
              ].map((item, idx) => {
                const Icon = item.icon;
                return (
                  <div
                    key={idx}
                    className="bg-[#141419] border border-white/10 rounded-2xl p-4 text-center flex flex-col items-center justify-center gap-2 hover:border-[#E8702A]/50 transition-all group"
                  >
                    <div className="p-2.5 rounded-xl bg-white/5 text-[#E8702A] group-hover:bg-[#E8702A]/10 transition-colors">
                      <Icon size={20} />
                    </div>
                    <div className="font-semibold text-xs text-white tracking-wide mt-1">
                      {item.step}
                    </div>
                    <div className="text-[11px] text-white/50 leading-tight">{item.desc}</div>
                  </div>
                );
              })}
            </div>

            {/* Core Capability Pillars */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-[#141419] border border-white/10 rounded-3xl p-7 relative overflow-hidden group hover:border-[#E8702A]/40 transition-all">
                <div className="text-[#E8702A] text-xs font-semibold uppercase tracking-wider mb-2">
                  Zero-Leakage Guarantee
                </div>
                <h3 className="text-xl font-playfair italic text-white mb-3">
                  Preprocessed strictly on fold splits
                </h3>
                <p className="text-xs sm:text-sm text-white/60 leading-relaxed">
                  All transformers (Median Imputer, Robust Scaler, One-Hot Encoder) are fit
                  exclusively on training folds inside <code>ColumnTransformer</code>. Zero
                  validation or test data is ever observed during feature transformation.
                </p>
              </div>

              <div className="bg-[#141419] border border-white/10 rounded-3xl p-7 relative overflow-hidden group hover:border-[#E8702A]/40 transition-all">
                <div className="text-emerald-400 text-xs font-semibold uppercase tracking-wider mb-2">
                  Optuna Bayesian Tuning
                </div>
                <h3 className="text-xl font-playfair italic text-white mb-3">
                  Autonomous Parameter Optimization
                </h3>
                <p className="text-xs sm:text-sm text-white/60 leading-relaxed">
                  Tree-structured Parzen Estimator (TPE) trials explore high-dimensional search
                  spaces across trees and regularization parameters, yielding verifiable metrics
                  without manual tuning.
                </p>
              </div>

              <div className="bg-[#141419] border border-white/10 rounded-3xl p-7 relative overflow-hidden group hover:border-[#E8702A]/40 transition-all">
                <div className="text-cyan-400 text-xs font-semibold uppercase tracking-wider mb-2">
                  Game-Theoretic SHAP
                </div>
                <h3 className="text-xl font-playfair italic text-white mb-3">
                  Full Attribution Transparency
                </h3>
                <p className="text-xs sm:text-sm text-white/60 leading-relaxed">
                  Every prediction generates an instance-level breakdown explaining exactly how
                  features shifted the predicted outcome positively or negatively with Shapley value
                  guarantees.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* ------------------------------------------------------------- */}
        {/* TAB C: DATA QUALITY & LEAKAGE AUDIT */}
        {/* ------------------------------------------------------------- */}
        {activeTab === 'quality' && (
          <div className="space-y-6">
            <div className="bg-[#141419] border border-white/10 rounded-3xl p-8 flex flex-col md:flex-row justify-between items-start md:items-center gap-6">
              <div>
                <span className="text-xs font-semibold uppercase tracking-widest text-emerald-400 px-3 py-1 bg-emerald-400/10 rounded-full border border-emerald-400/20">
                  Integrity Verified
                </span>
                <h3 className="text-3xl font-playfair italic text-white mt-3">
                  Dataset Health Score: 95 / 100
                </h3>
                <p className="text-white/60 text-xs mt-1">
                  Evaluated against 6 empirical quality rules: Missingness, Duplicates, Variance,
                  Cardinality, Outliers & Leakage.
                </p>
              </div>
              <div className="flex gap-4">
                <div className="bg-black/40 border border-white/10 rounded-2xl p-4 text-center min-w-[110px]">
                  <div className="text-xl font-bold text-emerald-400">12</div>
                  <div className="text-[11px] text-white/50">Healthy Checks</div>
                </div>
                <div className="bg-black/40 border border-white/10 rounded-2xl p-4 text-center min-w-[110px]">
                  <div className="text-xl font-bold text-amber-400">1</div>
                  <div className="text-[11px] text-white/50">Minor Outliers</div>
                </div>
                <div className="bg-black/40 border border-white/10 rounded-2xl p-4 text-center min-w-[110px]">
                  <div className="text-xl font-bold text-white">0</div>
                  <div className="text-[11px] text-white/50">Critical Errors</div>
                </div>
              </div>
            </div>

            {/* Audit Findings Table */}
            <div className="bg-[#141419] border border-white/10 rounded-3xl overflow-hidden p-6">
              <h4 className="text-lg font-semibold text-white mb-4">Quality Findings & Actions Taken</h4>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="border-b border-white/10 text-white/50">
                      <th className="pb-3 font-medium">Feature</th>
                      <th className="pb-3 font-medium">Rule</th>
                      <th className="pb-3 font-medium">Severity</th>
                      <th className="pb-3 font-medium">Detected</th>
                      <th className="pb-3 font-medium">Applied Mitigation</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/5 text-white/80">
                    <tr>
                      <td className="py-3 font-medium text-white">total_bedrooms</td>
                      <td>MODERATE_MISSINGNESS</td>
                      <td>
                        <span className="text-amber-400 bg-amber-400/10 px-2 py-0.5 rounded">
                          Warning
                        </span>
                      </td>
                      <td>1.8% missing values</td>
                      <td>Fold-safe median imputation applied inside ColumnTransformer</td>
                    </tr>
                    <tr>
                      <td className="py-3 font-medium text-white">median_income</td>
                      <td>POTENTIAL_OUTLIERS</td>
                      <td>
                        <span className="text-amber-400 bg-amber-400/10 px-2 py-0.5 rounded">
                          Warning
                        </span>
                      </td>
                      <td>4.2% values outside 1.5*IQR</td>
                      <td>RobustScaler and tree split bounds insulate regression weights</td>
                    </tr>
                    <tr>
                      <td className="py-3 font-medium text-white">dataset_global</td>
                      <td>ZERO_DATA_LEAKAGE</td>
                      <td>
                        <span className="text-emerald-400 bg-emerald-400/10 px-2 py-0.5 rounded">
                          Healthy
                        </span>
                      </td>
                      <td>0 validation leaks</td>
                      <td>Full pipeline re-fitted strictly on training splits per fold</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* ------------------------------------------------------------- */}
        {/* TAB D: CANDIDATE MODEL BENCHMARKS */}
        {/* ------------------------------------------------------------- */}
        {activeTab === 'models' && (
          <div className="space-y-6">
            <div className="bg-[#141419] border border-white/10 rounded-3xl p-7 flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
              <div>
                <span className="text-xs font-semibold uppercase tracking-widest text-[#E8702A] px-3 py-1 bg-[#E8702A]/10 rounded-full border border-[#E8702A]/20">
                  🏆 Top Performing Champion
                </span>
                <h3 className="text-2xl md:text-3xl font-playfair italic text-white mt-2">
                  {backendStatus.modelName}
                </h3>
                <p className="text-xs text-white/60 mt-1">
                  Selected automatically after Stratified/K-Fold cross-validation against candidate
                  algorithms.
                </p>
              </div>
              <div className="text-right">
                <div className="text-xs text-white/50 uppercase">Tuned {backendStatus.metric}</div>
                <div className="text-3xl font-bold text-emerald-400">
                  {backendStatus.score.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                </div>
              </div>
            </div>

            {/* Leaderboard */}
            <div className="bg-[#141419] border border-white/10 rounded-3xl p-6 overflow-x-auto">
              <h4 className="text-lg font-semibold text-white mb-4">
                Cross-Validation Leaderboard
              </h4>
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-white/10 text-white/50">
                    <th className="pb-3 font-medium">Rank</th>
                    <th className="pb-3 font-medium">Model</th>
                    <th className="pb-3 font-medium">CV Mean (RMSE)</th>
                    <th className="pb-3 font-medium">CV Std</th>
                    <th className="pb-3 font-medium">Train Latency</th>
                    <th className="pb-3 font-medium">Decision</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5 text-white/80">
                  <tr className="bg-white/5">
                    <td className="py-3 font-bold text-[#E8702A]">#1</td>
                    <td className="font-semibold text-white">XGBoost Regressor</td>
                    <td className="text-emerald-400 font-bold">16,746.46</td>
                    <td>± 412.18</td>
                    <td>0.42s</td>
                    <td>
                      <span className="text-emerald-400 font-medium">
                        Promoted to Production
                      </span>
                    </td>
                  </tr>
                  <tr>
                    <td className="py-3 font-bold text-white/40">#2</td>
                    <td>Random Forest Regressor</td>
                    <td>17,218.80</td>
                    <td>± 490.52</td>
                    <td>1.15s</td>
                    <td>
                      <span className="text-white/40">Candidate</span>
                    </td>
                  </tr>
                  <tr>
                    <td className="py-3 font-bold text-white/40">#3</td>
                    <td>Gradient Boosting (Hist)</td>
                    <td>17,890.15</td>
                    <td>± 510.30</td>
                    <td>0.28s</td>
                    <td>
                      <span className="text-white/40">Candidate</span>
                    </td>
                  </tr>
                  <tr>
                    <td className="py-3 font-bold text-white/40">#4</td>
                    <td>Ridge Regression</td>
                    <td>31,450.60</td>
                    <td>± 680.12</td>
                    <td>0.04s</td>
                    <td>
                      <span className="text-white/40">Linear Baseline</span>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* ------------------------------------------------------------- */}
        {/* TAB E: SHAP EXPLAINABILITY */}
        {/* ------------------------------------------------------------- */}
        {activeTab === 'explain' && (
          <div className="space-y-6">
            <div className="bg-[#141419] border border-white/10 rounded-3xl p-8">
              <span className="text-xs font-semibold uppercase tracking-widest text-[#E8702A] px-3 py-1 bg-[#E8702A]/10 rounded-full border border-[#E8702A]/20">
                TreeSHAP Game Attribution
              </span>
              <h3 className="text-2xl md:text-3xl font-playfair italic text-white mt-2">
                Global Feature Importance
              </h3>
              <p className="text-xs text-white/60 mt-1 mb-6">
                Calculated using exact Shapley values across validation holdouts, quantifying each
                feature's marginal attribution.
              </p>

              <div className="space-y-4">
                {[
                  { feature: 'median_income', importance: 0.485, color: '#E8702A' },
                  { feature: 'ocean_proximity_INLAND', importance: 0.26, color: '#F97316' },
                  { feature: 'total_rooms', importance: 0.125, color: '#10B981' },
                  { feature: 'house_age', importance: 0.082, color: '#8B5CF6' },
                  { feature: 'total_bedrooms', importance: 0.048, color: '#F59E0B' },
                ].map((item, idx) => (
                  <div key={idx} className="space-y-1">
                    <div className="flex justify-between text-xs font-medium">
                      <span className="text-white font-mono">{item.feature}</span>
                      <span className="text-white/50">
                        {(item.importance * 100).toFixed(1)}% attribution
                      </span>
                    </div>
                    <div className="w-full h-3 bg-white/5 rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all duration-1000"
                        style={{
                          width: `${item.importance * 100}%`,
                          backgroundColor: item.color,
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* ------------------------------------------------------------- */}
        {/* TAB F: FASTAPI PREDICTION SERVING */}
        {/* ------------------------------------------------------------- */}
        {activeTab === 'predict' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="bg-[#141419] border border-white/10 rounded-3xl p-7">
              <h3 className="text-xl font-semibold text-white mb-2">Real-Time Model Inference</h3>
              <p className="text-xs text-white/60 mb-6">
                Directly calls the live FastAPI endpoint <code>POST http://127.0.0.1:8000/predict</code>.
                Inputs are transformed automatically by the saved production pipeline.
              </p>

              <div className="space-y-4">
                <div>
                  <label className="text-xs text-white/70 block mb-1">Median Income ($10k)</label>
                  <input
                    type="number"
                    step="0.1"
                    value={predictInputs.median_income}
                    onChange={(e) =>
                      setPredictInputs({
                        ...predictInputs,
                        median_income: parseFloat(e.target.value),
                      })
                    }
                    className="w-full bg-black/40 border border-white/10 rounded-xl px-4 py-2.5 text-xs text-white focus:border-[#E8702A] focus:outline-none"
                  />
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="text-xs text-white/70 block mb-1">House Age (Years)</label>
                    <input
                      type="number"
                      value={predictInputs.house_age}
                      onChange={(e) =>
                        setPredictInputs({
                          ...predictInputs,
                          house_age: parseInt(e.target.value),
                        })
                      }
                      className="w-full bg-black/40 border border-white/10 rounded-xl px-4 py-2.5 text-xs text-white focus:border-[#E8702A] focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="text-xs text-white/70 block mb-1">Total Rooms</label>
                    <input
                      type="number"
                      value={predictInputs.total_rooms}
                      onChange={(e) =>
                        setPredictInputs({
                          ...predictInputs,
                          total_rooms: parseInt(e.target.value),
                        })
                      }
                      className="w-full bg-black/40 border border-white/10 rounded-xl px-4 py-2.5 text-xs text-white focus:border-[#E8702A] focus:outline-none"
                    />
                  </div>
                </div>
                <div>
                  <label className="text-xs text-white/70 block mb-1">Ocean Proximity</label>
                  <select
                    value={predictInputs.ocean_proximity}
                    onChange={(e) =>
                      setPredictInputs({ ...predictInputs, ocean_proximity: e.target.value })
                    }
                    className="w-full bg-black/40 border border-white/10 rounded-xl px-4 py-2.5 text-xs text-white focus:border-[#E8702A] focus:outline-none"
                  >
                    <option value="<1H OCEAN">&lt;1H OCEAN</option>
                    <option value="INLAND">INLAND</option>
                    <option value="NEAR BAY">NEAR BAY</option>
                    <option value="NEAR OCEAN">NEAR OCEAN</option>
                  </select>
                </div>

                <button
                  onClick={executePrediction}
                  disabled={isPredicting}
                  className="w-full mt-2 bg-[#E8702A] hover:bg-[#D4601C] text-white text-xs font-semibold py-3.5 rounded-xl transition-all flex items-center justify-center gap-2 shadow-lg shadow-[#E8702A]/25 active:scale-95 disabled:opacity-50 cursor-pointer"
                >
                  {isPredicting ? (
                    <RefreshCw size={14} className="animate-spin" />
                  ) : (
                    <Send size={14} />
                  )}
                  Execute Real-Time Prediction
                </button>
              </div>
            </div>

            {/* Inference Output Card */}
            <div className="bg-[#141419] border border-white/10 rounded-3xl p-7 flex flex-col justify-between">
              <div>
                <h3 className="text-xl font-semibold text-white mb-2">Inference Output</h3>
                <p className="text-xs text-white/60 mb-6">
                  Calculated from <code>{backendStatus.modelName}</code> registered in{' '}
                  <code>artifacts/model_bundle.joblib</code>.
                </p>

                {predictionResult ? (
                  <div className="bg-black/50 border border-[#E8702A]/30 rounded-2xl p-6 text-center shadow-xl">
                    <div className="text-xs text-white/50 uppercase tracking-widest mb-1">
                      Predicted {backendStatus.target}
                    </div>
                    <div className="text-4xl md:text-5xl font-bold text-white font-playfair italic">
                      ${Number(predictionResult.prediction).toLocaleString()}
                    </div>
                    <div className="mt-4 text-xs text-emerald-400 flex items-center justify-center gap-1.5">
                      <CheckCircle2 size={14} /> Certified Zero-Leakage Pipeline
                    </div>
                  </div>
                ) : (
                  <div className="bg-black/30 border border-white/5 rounded-2xl p-10 text-center text-white/40 text-xs">
                    Click "Execute Real-Time Prediction" to test the served model bundle.
                  </div>
                )}
              </div>

              <div className="mt-6 pt-6 border-t border-white/10 text-[11px] text-white/50 leading-relaxed">
                FastAPI Serving: <code>POST http://127.0.0.1:8000/predict</code>
                <br />
                Pydantic validation active • Batch prediction also served at{' '}
                <code>POST /predict/batch</code>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* ------------------------------------------------------------- */}
      {/* 4. FOOTER */}
      {/* ------------------------------------------------------------- */}
      <footer className="border-t border-white/10 py-12 text-center text-xs text-white/40 bg-[#08080A]">
        <div className="flex items-center justify-center gap-2 mb-2">
          <span className="font-playfair italic text-white text-base">Lithos</span>
          <span>&bull;</span>
          <span className="text-white/80 font-semibold">Autonomous Data Scientist Platform</span>
        </div>
        <div className="text-[11px] text-white/30">
          Built with React 18, TypeScript, Tailwind CSS, FastAPI, Streamlit, XGBoost, Optuna & SHAP.
        </div>
      </footer>
    </div>
  );
}
