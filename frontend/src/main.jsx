import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { DrawingUtils, FilesetResolver, PoseLandmarker } from '@mediapipe/tasks-vision'
import { createRoot } from 'react-dom/client'
import { Activity, AlertTriangle, ArrowDownRight, ArrowUpRight, BarChart3, Bell, BrainCircuit, CheckCircle2, ChevronDown, CircleHelp, Clock3, Download, FileVideo, Filter, Gauge, LayoutDashboard, LoaderCircle, Menu, Play, Radio, RefreshCw, Search, ShieldCheck, Sparkles, Trash2, Upload, Video, Webcam, X } from 'lucide-react'
import { Area, AreaChart, CartesianGrid, Pie, PieChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis, BarChart, Bar } from 'recharts'
import './styles.css'

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
const COLORS = ['#8b7cff', '#4c9aff', '#35d6c4', '#ffb86b', '#f879a8', '#f16d78']
const nav = [
  { id: 'overview', label: 'Overview', icon: LayoutDashboard, group: 'WORKSPACE' },
  { id: 'live', label: 'Live detection', icon: Radio, group: 'WORKSPACE' },
  { id: 'analyze', label: 'Video analysis', icon: FileVideo, group: 'WORKSPACE' },
  { id: 'history', label: 'Activity history', icon: Clock3, group: 'INSIGHTS' },
  { id: 'model', label: 'Model performance', icon: BarChart3, group: 'INSIGHTS' },
]
const pretty = (v = '') => String(v).replaceAll('_', ' ')
const dateLabel = (v) => { if (!v) return '—'; const d = new Date(v.replace(' ', 'T')); return Number.isNaN(d.getTime()) ? v : d.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) }
async function api(path, options = {}) {
  const res = await fetch(`${API}${path}`, options)
  const data = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`)
  return data
}
function App() {
  const [page, setPage] = useState('overview')
  const [dashboard, setDashboard] = useState(null)
  const [history, setHistory] = useState([])
  const [model, setModel] = useState(null)
  const [metrics, setMetrics] = useState(null)
  const [samples, setSamples] = useState([])
  const [health, setHealth] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [mobileNav, setMobileNav] = useState(false)
  const [search, setSearch] = useState('')
  const [activityFilter, setActivityFilter] = useState('all')
  const [fallsOnly, setFallsOnly] = useState(false)
  const [result, setResult] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [file, setFile] = useState(null)
  const [selectedSample, setSelectedSample] = useState('')
  const [webcamOn, setWebcamOn] = useState(false)
  const [frameResult, setFrameResult] = useState(null)
  const [cameraError, setCameraError] = useState('')
  const [clearing, setClearing] = useState(false)
  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const canvasRef = useRef(null)
  const poseCanvasRef = useRef(null)
  const poseLandmarkerRef = useRef(null)
  const poseAnimationRef = useRef(null)
  const [poseStatus, setPoseStatus] = useState('idle')
  const fileRef = useRef(null)
  const [toast, setToast] = useState('')

  const refresh = useCallback(async (initial = false) => {
    if (initial) setLoading(true)
    try {
      const [d, h, m, met, s] = await Promise.all([api('/api/dashboard'), api('/api/health'), api('/api/model'), api('/api/metrics'), api('/api/samples')])
      setDashboard(d); setHealth(h); setModel(m); setMetrics(met); setSamples(s.items || []); setError('')
    } catch (e) { setError(e.message); setHealth({ status: 'offline' }) }
    try { const h = await api(`/api/history?limit=200&search=${encodeURIComponent(search)}&activity=${encodeURIComponent(activityFilter)}&falls_only=${fallsOnly}`); setHistory(h.items || []) } catch (e) { if (!error) setError(e.message) }
    if (initial) setLoading(false)
  }, [search, activityFilter, fallsOnly])
  useEffect(() => { refresh(true) }, [])
  useEffect(() => { const t = setInterval(() => refresh(false), 15000); return () => clearInterval(t) }, [refresh])
  useEffect(() => { if (toast) { const t = setTimeout(() => setToast(''), 3000); return () => clearTimeout(t) } }, [toast])
  useEffect(() => { if (page === 'history') refresh(false) }, [page, search, activityFilter, fallsOnly])
  // Attach the stream after React has rendered the <video> element.
  // startCamera() previously checked videoRef before that element existed.
  useEffect(() => {
    const video = videoRef.current
    const stream = streamRef.current
    if (webcamOn && video && stream) {
      video.srcObject = stream
      video.play().catch(() => setCameraError('Camera opened, but the browser could not play the video. Try refreshing the page.'))
    }
  }, [webcamOn])
  useEffect(() => () => stopCamera(), [])

  // Live pose tracking: draw MediaPipe's body landmarks on a transparent canvas above the webcam.
  useEffect(() => {
    if (!webcamOn) {
      if (poseAnimationRef.current) cancelAnimationFrame(poseAnimationRef.current)
      poseAnimationRef.current = null
      if (poseLandmarkerRef.current) poseLandmarkerRef.current.close()
      poseLandmarkerRef.current = null
      const canvas = poseCanvasRef.current
      if (canvas) canvas.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height)
      setPoseStatus('idle')
      return
    }

    let cancelled = false
    let landmarker = null
    let lastVideoTime = -1

    async function initialisePose() {
      setPoseStatus('loading')
      try {
        const vision = await FilesetResolver.forVisionTasks(
          '/mediapipe/wasm'
        )
        const options = {
          baseOptions: {
            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
            delegate: 'GPU'
          },
          runningMode: 'VIDEO',
          numPoses: 1
        }
        try {
          landmarker = await PoseLandmarker.createFromOptions(vision, options)
        } catch {
          // CPU fallback for computers/browsers where GPU delegate is unavailable.
          landmarker = await PoseLandmarker.createFromOptions(vision, {
            ...options,
            baseOptions: { modelAssetPath: options.baseOptions.modelAssetPath, delegate: 'CPU' }
          })
        }
        if (cancelled) { landmarker.close(); return }
        poseLandmarkerRef.current = landmarker
        setPoseStatus('tracking')

        const drawPose = () => {
          if (cancelled) return
          const video = videoRef.current
          const canvas = poseCanvasRef.current
          if (video && canvas && video.readyState >= 2 && video.videoWidth > 0) {
            if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
              canvas.width = video.videoWidth
              canvas.height = video.videoHeight
            }
            const ctx = canvas.getContext('2d')
            if (video.currentTime !== lastVideoTime) {
              lastVideoTime = video.currentTime
              ctx.clearRect(0, 0, canvas.width, canvas.height)
              const result = landmarker.detectForVideo(video, performance.now())
              if (result.landmarks?.length) {
                const drawer = new DrawingUtils(ctx)
                for (const landmarks of result.landmarks) {
                  drawer.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
                    color: '#35d6c4', lineWidth: 4
                  })
                  drawer.drawLandmarks(landmarks, {
                    color: '#ffb86b', fillColor: '#8b7cff', lineWidth: 2, radius: 4
                  })
                }
              }
            }
          }
          poseAnimationRef.current = requestAnimationFrame(drawPose)
        }
        drawPose()
      } catch (e) {
        if (!cancelled) {
          console.error('MediaPipe pose initialisation failed:', e)
          setPoseStatus('error')
          setCameraError('Skeleton model could not load. Check your internet connection, then refresh the page and start the camera again.')
        }
      }
    }

    initialisePose()
    return () => {
      cancelled = true
      if (poseAnimationRef.current) cancelAnimationFrame(poseAnimationRef.current)
      poseAnimationRef.current = null
      if (landmarker) landmarker.close()
      if (poseLandmarkerRef.current === landmarker) poseLandmarkerRef.current = null
    }
  }, [webcamOn])

  function stopCamera() {
    if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop())
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    setWebcamOn(false)
  }
  async function startCamera() {
    setCameraError('')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false })
      streamRef.current = stream
      setWebcamOn(true)
    } catch (e) { setCameraError(e.name === 'NotAllowedError' ? 'Camera permission was denied. Allow camera access in your browser settings.' : 'Could not open the camera. Check that a webcam is connected and not used by another app.') }
  }
  async function captureFrame() {
    const video = videoRef.current, canvas = canvasRef.current
    if (!video || !canvas || !video.videoWidth) return
    canvas.width = video.videoWidth; canvas.height = video.videoHeight
    canvas.getContext('2d').drawImage(video, 0, 0)
    try { const data = await api('/api/predict-frame', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ image: canvas.toDataURL('image/jpeg', .8), source: 'Browser Webcam' }) }); setFrameResult(data) }
    catch (e) { setCameraError(e.message) }
  }
  function chooseFile(f) { if (!f) return; setFile(f); setResult(null); setSelectedSample('') }
  async function analyze() {
    if (!file && !selectedSample) { setToast('Choose a video or one of the sample clips first.'); return }
    setUploading(true); setResult(null); setError('')
    try {
      let response
      if (selectedSample) {
        const res = await fetch(`${API}${selectedSample}`); if (!res.ok) throw new Error('Could not load selected sample video.')
        const blob = await res.blob(); const name = selectedSample.split('/').pop() || 'sample.mp4'
        const fd = new FormData(); fd.append('file', new File([blob], name, { type: 'video/mp4' }))
        response = await api('/api/analyze', { method: 'POST', body: fd })
      } else { const fd = new FormData(); fd.append('file', file); response = await api('/api/analyze', { method: 'POST', body: fd }) }
      setResult(response); setToast('Video analysis completed.'); await refresh(false)
    } catch (e) { setError(e.message) } finally { setUploading(false) }
  }
  async function clearHistory() {
    if (!window.confirm('Delete all activity history? This cannot be undone.')) return
    setClearing(true); try { await api('/api/history', { method: 'DELETE' }); await refresh(false); setToast('Activity history cleared.') } catch (e) { setError(e.message) } finally { setClearing(false) }
  }
  function exportCsv() {
    const rows = history
    const keys = ['id', 'timestamp', 'source', 'activity', 'confidence', 'is_fall', 'duration_seconds', 'details']
    const csv = [keys.join(','), ...rows.map(r => keys.map(k => `"${String(r[k] ?? '').replaceAll('"', '""')}"`).join(','))].join('\n')
    const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8;' })); const a = document.createElement('a'); a.href = url; a.download = 'motionai-activity-history.csv'; a.click(); URL.revokeObjectURL(url)
  }
  const dist = dashboard?.activity_distribution?.length ? dashboard.activity_distribution : []
  const trend = useMemo(() => (history || []).slice(0, 12).reverse().map((r, i) => ({ time: r.timestamp?.slice(11, 16) || `#${i + 1}`, confidence: Number(r.confidence) || 0 })), [history])
  const activeNav = nav.find(n => n.id === page)
  const displayError = error && health?.status === 'offline'

  return <div className="app-shell">
    <aside className={`sidebar ${mobileNav ? 'sidebar-open' : ''}`}>
      <div className="brand"><div className="brand-mark"><Activity size={22}/></div><div><div className="brand-name">motion<span>ai</span></div><div className="brand-sub">ACTIVITY INTELLIGENCE</div></div><button className="icon-btn mobile-close" onClick={() => setMobileNav(false)} aria-label="Close menu"><X size={18}/></button></div>
      <div className="workspace-switch"><div className="workspace-avatar">M</div><div className="workspace-copy"><strong>My workspace</strong><span>Personal project</span></div><ChevronDown size={15} className="muted"/></div>
      {['WORKSPACE', 'INSIGHTS'].map(group => <div className="nav-group" key={group}><div className="nav-label">{group}</div>{nav.filter(n => n.group === group).map(n => { const Icon = n.icon; return <button key={n.id} className={`nav-item ${page === n.id ? 'active' : ''}`} onClick={() => { setPage(n.id); setMobileNav(false) }}><Icon size={18}/><span>{n.label}</span>{n.id === 'live' && <i className="live-dot"/>}</button> })}</div>)}
      <div className="sidebar-spacer"/><div className="sidebar-bottom"><div className="model-mini"><div className="mini-icon"><BrainCircuit size={18}/></div><div className="model-mini-copy"><strong>Activity model</strong><span><i className={`status-dot ${health?.model_file ? 'online' : ''}`}/>{health?.model_file ? 'Model loaded' : 'Model unavailable'}</span></div><button className="icon-btn" onClick={() => refresh(false)} title="Refresh status"><RefreshCw size={14}/></button></div><button className="nav-item help-item" onClick={() => setPage('model')}><CircleHelp size={17}/><span>Help & documentation</span></button><div className="user-card"><div className="user-avatar">N</div><div className="user-info"><strong>Project workspace</strong><span>Local development</span></div><span className="user-menu">•••</span></div></div>
    </aside>
    {mobileNav && <button className="mobile-backdrop" aria-label="Close navigation" onClick={() => setMobileNav(false)}/>}
    <main className="main-area">
      <header className="topbar"><div className="topbar-left"><button className="icon-btn menu-btn" onClick={() => setMobileNav(true)} aria-label="Open menu"><Menu size={20}/></button><div className="breadcrumb">Workspace <span>/</span> <strong>{activeNav?.label || 'Overview'}</strong></div></div><div className="topbar-actions"><div className={`api-status ${health?.status === 'ok' ? 'connected' : 'disconnected'}`}><span className="status-dot"/>{health?.status === 'ok' ? 'API connected' : 'API offline'}</div><button className="icon-btn notification-btn" title="Notifications" onClick={() => setToast(dashboard?.fall_alerts ? `${dashboard.fall_alerts} fall alert(s) in saved history.` : 'No saved fall alerts.')}><Bell size={18}/>{dashboard?.fall_alerts > 0 && <i/>}</button><div className="top-avatar">N</div></div></header>
      <div className="page-wrap">
        {displayError && <div className="alert alert-error"><AlertTriangle size={18}/><div><strong>Backend unavailable</strong><p>{error}. Start the FastAPI server using the README instructions.</p></div><button className="icon-btn" onClick={() => refresh(true)}><RefreshCw size={16}/></button></div>}
        {page === 'overview' && <>
          <div className="page-heading"><div><div className="eyebrow"><Sparkles size={14}/> ACTIVITY INTELLIGENCE</div><h1>Good evening<span className="heading-dot">.</span></h1><p className="page-description">Monitor human movement, review activity patterns, and keep an eye on safety signals.</p></div><div className="heading-actions"><span className="date-chip"><Clock3 size={15}/>{new Date().toLocaleDateString([], { month: 'short', day: 'numeric', year: 'numeric' })}</span><button className="btn btn-primary" onClick={() => { setPage('analyze'); setResult(null) }}><Upload size={16}/> Analyze a video</button></div></div>
          <div className="metric-grid">
            <Metric icon={Activity} label="Total predictions" value={dashboard?.total_predictions ?? '—'} sub="Saved in activity history" accent="violet" />
            <Metric icon={ShieldCheck} label="Average confidence" value={dashboard ? `${dashboard.avg_confidence}%` : '—'} sub="Across recorded predictions" accent="blue" />
            <Metric icon={AlertTriangle} label="Fall alerts" value={dashboard?.fall_alerts ?? '—'} sub="Flagged events in history" accent="rose" />
            <Metric icon={BrainCircuit} label="Activity classes" value={model?.class_count ?? '—'} sub="Classes in the model" accent="mint" />
          </div>
          <div className="content-grid overview-grid"><section className="panel activity-panel"><PanelTitle title="Activity distribution" subtitle="Prediction counts grouped by activity" action={<span className="small-badge"><span className="status-dot online"/> Live data</span>}/>{dist.length ? <div className="chart-layout"><div className="donut-wrap"><ResponsiveContainer width="100%" height={220}><PieChart><Pie data={dist} dataKey="count" nameKey="label" innerRadius={64} outerRadius={88} paddingAngle={4} stroke="none">{dist.map((d, i) => <Cell key={d.activity} fill={COLORS[i % COLORS.length]}/>)}</Pie><Tooltip contentStyle={tooltipStyle}/></PieChart></ResponsiveContainer><div className="donut-center"><strong>{dist.reduce((a, b) => a + b.count, 0)}</strong><span>predictions</span></div></div><div className="legend-list">{dist.map((d, i) => <div className="legend-row" key={d.activity}><span className="legend-swatch" style={{ background: COLORS[i % COLORS.length] }}/><span className="legend-name">{d.label}</span><strong>{d.count}</strong></div>)}</div></div> : <EmptyState icon={BarChart3} title="Your activity insights start here" text="Analyze a video to populate the distribution chart with real predictions." action={() => setPage('analyze')} actionText="Analyze video"/>}</section>
            <section className="panel confidence-panel"><PanelTitle title="Confidence trend" subtitle="Recent saved prediction confidence" action={<span className="tiny-pill">Recent 12</span>}/>{trend.length ? <div className="trend-chart"><ResponsiveContainer width="100%" height={228}><AreaChart data={trend} margin={{ top: 12, right: 4, left: -22, bottom: 0 }}><defs><linearGradient id="confidenceGradient" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#8b7cff" stopOpacity={0.32}/><stop offset="95%" stopColor="#8b7cff" stopOpacity={0}/></linearGradient></defs><CartesianGrid strokeDasharray="3 4" stroke="#263047" vertical={false}/><XAxis dataKey="time" tick={{ fill: '#77839c', fontSize: 11 }} axisLine={false} tickLine={false}/><YAxis domain={[0, 100]} tick={{ fill: '#77839c', fontSize: 11 }} axisLine={false} tickLine={false}/><Tooltip contentStyle={tooltipStyle} formatter={(v) => [`${v}%`, 'Confidence']}/><Area type="monotone" dataKey="confidence" stroke="#9b8cff" strokeWidth={2.5} fill="url(#confidenceGradient)"/></AreaChart></ResponsiveContainer></div> : <EmptyState icon={Gauge} title="No confidence trend yet" text="The chart appears once predictions are saved." action={() => setPage('analyze')} actionText="Run first analysis"/>}</section>
          </div>
          <section className="panel recent-panel"><PanelTitle title="Recent activity" subtitle="Latest entries from your local database" action={<button className="text-action" onClick={() => setPage('history')}>View all <ArrowUpRight size={15}/></button>}/><HistoryTable rows={dashboard?.recent || []} compact onEmpty={() => setPage('analyze')}/></section>
          <div className="bottom-grid"><div className="insight-card"><div className="insight-icon"><BrainCircuit size={20}/></div><div><strong>Spatio-temporal AI model</strong><p>{model?.name || 'LRCN (CNN + BiLSTM)'} analyzes a sequence of frames to classify movement.</p></div><button className="icon-btn" onClick={() => setPage('model')}><ArrowUpRight size={17}/></button></div><div className="insight-card safety-card"><div className="insight-icon safety-icon"><ShieldCheck size={20}/></div><div><strong>Safety signal monitoring</strong><p>Fall predictions are indicators for review, not a substitute for emergency or medical systems.</p></div><button className="icon-btn" onClick={() => { setPage('history'); setFallsOnly(true) }}><ArrowUpRight size={17}/></button></div></div>
        </>}
        {page === 'analyze' && <><div className="page-heading"><div><div className="eyebrow"><FileVideo size={14}/> VIDEO INTELLIGENCE</div><h1>Video analysis<span className="heading-dot">.</span></h1><p className="page-description">Upload a clip and let the LRCN model estimate the activity across sampled frames.</p></div><span className="date-chip"><ShieldCheck size={15}/> Local processing</span></div><div className="analysis-layout"><section className="panel upload-panel"><PanelTitle title="Input video" subtitle="MP4, AVI, MOV, MKV or WEBM · max 100 MB"/><div className={`dropzone ${dragging ? 'dragging' : ''} ${file ? 'has-file' : ''}`} onDragOver={e => { e.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={e => { e.preventDefault(); setDragging(false); chooseFile(e.dataTransfer.files?.[0]) }} onClick={() => fileRef.current?.click()} role="button" tabIndex={0} onKeyDown={e => e.key === 'Enter' && fileRef.current?.click()}><input ref={fileRef} type="file" accept="video/mp4,video/avi,video/quicktime,video/x-matroska,video/webm" hidden onClick={e => e.stopPropagation()} onChange={e => chooseFile(e.target.files?.[0])}/><div className="upload-symbol"><Upload size={23}/></div><strong>{file ? file.name : 'Drop your video here'}</strong><p>{file ? `${(file.size / (1024 * 1024)).toFixed(2)} MB · Ready to analyze` : 'or click to browse files from your computer'}</p><span className="browse-btn">Browse files</span></div>{file && <div className="selected-file"><div className="file-icon"><Video size={19}/></div><div className="file-copy"><strong>{file.name}</strong><span>{(file.size / (1024 * 1024)).toFixed(2)} MB</span></div><button className="icon-btn" onClick={() => { setFile(null); setResult(null) }} aria-label="Remove file"><X size={16}/></button></div>}<div className="or-divider"><span/>OR TRY A SAMPLE CLIP<span/></div><div className="sample-grid">{samples.map(s => <button key={s.name} className={`sample-chip ${selectedSample === s.url ? 'selected' : ''}`} onClick={() => { setSelectedSample(s.url); setFile(null); setResult(null) }}><Play size={13}/>{s.label}</button>)}</div><button className="btn btn-primary btn-wide" disabled={uploading || (!file && !selectedSample)} onClick={analyze}>{uploading ? <><LoaderCircle className="spin" size={17}/> Analyzing frames…</> : <><Sparkles size={17}/> Analyze activity</>}</button>{uploading && <div className="processing-note"><span className="processing-line"/>Extracting frames, running inference, and saving the result.</div>}</section><section className="panel result-panel"><PanelTitle title="Analysis result" subtitle="Your prediction will appear here"/>{result ? <div className="result-content"><div className={`result-hero ${result.is_fall ? 'fall-result' : ''}`}><div className="result-icon">{result.is_fall ? <AlertTriangle size={24}/> : <Activity size={24}/>}</div><div><span className="result-label">PREDICTED ACTIVITY</span><h2>{pretty(result.label || result.activity)}</h2><p>{result.is_fall ? 'Potential fall pattern detected — please review the clip.' : 'Top predicted class from the selected video.'}</p></div><span className={`confidence-chip ${result.is_fall ? 'danger' : ''}`}>{result.confidence}%</span></div><div className="result-meta"><div><span>Video</span><strong>{result.filename}</strong></div><div><span>Duration</span><strong>{result.duration_seconds || 0}s</strong></div><div><span>Model</span><strong>LRCN</strong></div></div><div className="probability-title"><strong>Class probabilities</strong><span>Model confidence</span></div><div className="probability-list">{(result.probabilities || []).map((p, i) => <div className="prob-row" key={p.activity}><div className="prob-label"><span>{pretty(p.label)}</span><strong>{p.confidence}%</strong></div><div className="prob-track"><span style={{ width: `${p.confidence}%`, background: p.activity === 'Falling' ? '#f16d78' : COLORS[i % COLORS.length] }}/></div></div>)}</div><div className="disclaimer"><ShieldCheck size={16}/><span>Predictions are model estimates and may be wrong. This demo is not a certified fall alarm system.</span></div><button className="btn btn-secondary btn-wide" onClick={() => { setResult(null); setFile(null); setSelectedSample('') }}>Analyze another video</button></div> : <div className="result-empty"><div className="result-orbit"><div className="orbit-core"><BrainCircuit size={33}/></div><i/><b/></div><h3>Waiting for your video</h3><p>Choose a video or sample clip. We'll extract 16 evenly spaced frames and run the trained model.</p><div className="pipeline"><div><span>01</span><strong>Frame sampling</strong></div><ArrowDownRight size={15}/><div><span>02</span><strong>CNN + BiLSTM</strong></div><ArrowDownRight size={15}/><div><span>03</span><strong>Prediction</strong></div></div></div>}</section></div></>}
        {page === 'live' && <><div className="page-heading"><div><div className="eyebrow"><Radio size={14}/> REAL-TIME PREVIEW</div><h1>Live detection<span className="heading-dot">.</span></h1><p className="page-description">Use your browser camera for a single-frame model preview. For motion-aware predictions, analyze a recorded video.</p></div><span className={`api-status ${webcamOn ? 'connected' : ''}`}><span className="status-dot"/>{webcamOn ? 'Camera active' : 'Camera idle'}</span></div><div className="live-layout"><section className="panel camera-panel"><PanelTitle title="Camera feed" subtitle="Browser webcam · permission required" action={webcamOn ? <span className="recording-pill"><i/> LIVE</span> : null}/><div className="camera-stage">{webcamOn ? <video ref={videoRef} autoPlay muted playsInline/> : <div className="camera-placeholder"><div className="camera-illustration"><Webcam size={42}/><span><i/><i/><i/></span></div><strong>Your camera feed will appear here</strong><p>Camera access stays in your browser. No video is recorded by this preview.</p></div>}{webcamOn && <canvas ref={poseCanvasRef} aria-label="Live body pose skeleton overlay" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}/>}<canvas ref={canvasRef} hidden/>{webcamOn && <div className="camera-overlay"><span><i/> LIVE PREVIEW</span><span>{poseStatus === 'tracking' ? 'Skeleton tracking active' : poseStatus === 'loading' ? 'Loading skeleton…' : poseStatus === 'error' ? 'Skeleton unavailable' : 'Browser camera'}</span></div>}</div><div className="camera-controls">{!webcamOn ? <button className="btn btn-primary" onClick={startCamera}><Webcam size={16}/> Start camera</button> : <><button className="btn btn-secondary" onClick={captureFrame}><Sparkles size={16}/> Classify current frame</button><button className="btn btn-danger" onClick={stopCamera}><X size={16}/> Stop camera</button></>}<span className="control-hint">{webcamOn ? 'Capture a still frame for model preview' : 'Camera permission is requested only after you click start'}</span></div>{cameraError && <div className="inline-error"><AlertTriangle size={15}/>{cameraError}</div>}</section><section className="panel live-result-panel"><PanelTitle title="Latest preview" subtitle="Frame-level classification"/>{frameResult ? <><div className="live-prediction"><span className="prediction-glow"><Activity size={22}/></span><div><span className="result-label">PREDICTED ACTIVITY</span><h2>{pretty(frameResult.label)}</h2><p>Confidence score</p></div><strong>{frameResult.confidence}%</strong></div><div className="confidence-meter"><span style={{ width: `${frameResult.confidence}%` }}/></div><div className="probability-list live-probs">{(frameResult.probabilities || []).slice(0, 4).map((p, i) => <div className="prob-row" key={p.activity}><div className="prob-label"><span>{pretty(p.label)}</span><strong>{p.confidence}%</strong></div><div className="prob-track"><span style={{ width: `${p.confidence}%`, background: COLORS[i] }}/></div></div>)}</div><div className="disclaimer"><AlertTriangle size={16}/><span>{frameResult.note}</span></div></> : <div className="empty-small"><div className="empty-icon"><Gauge size={24}/></div><strong>No preview result yet</strong><p>Start the camera and classify a frame to see the model output.</p></div>}<div className="model-facts"><div><span>Architecture</span><strong>CNN + BiLSTM</strong></div><div><span>Sequence input</span><strong>16 frames</strong></div><div><span>Inference device</span><strong>{health?.device || '—'}</strong></div></div></section></div></>}
        {page === 'history' && <><div className="page-heading"><div><div className="eyebrow"><Clock3 size={14}/> PREDICTION LOGS</div><h1>Activity history<span className="heading-dot">.</span></h1><p className="page-description">Search and review the predictions saved in your local SQLite database.</p></div><div className="heading-actions"><button className="btn btn-secondary" onClick={exportCsv}><Download size={16}/> Export CSV</button><button className="btn btn-danger-outline" onClick={clearHistory} disabled={clearing}>{clearing ? <LoaderCircle className="spin" size={16}/> : <Trash2 size={16}/>} Clear history</button></div></div><section className="panel history-panel"><div className="history-toolbar"><div className="search-field"><Search size={16}/><input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search source, activity, details…"/></div><div className="filter-control"><Filter size={15}/><select value={activityFilter} onChange={e => setActivityFilter(e.target.value)}><option value="all">All activities</option>{(model?.classes || []).map(c => <option key={c} value={c}>{pretty(c)}</option>)}</select></div><label className="toggle-filter"><input type="checkbox" checked={fallsOnly} onChange={e => setFallsOnly(e.target.checked)}/> Falls only</label></div><HistoryTable rows={history} onEmpty={() => setPage('analyze')}/><div className="table-footer"><span>Showing <strong>{history.length}</strong> records · newest first</span><button className="text-action" onClick={() => refresh(false)}><RefreshCw size={14}/> Refresh</button></div></section></>}
        {page === 'model' && <><div className="page-heading"><div><div className="eyebrow"><BrainCircuit size={14}/> MODEL INSIGHTS</div><h1>Model performance<span className="heading-dot">.</span></h1><p className="page-description">Understand the model pipeline, current evaluation artifacts, and what the scores can and cannot tell you.</p></div><span className="date-chip"><CheckCircle2 size={15}/> Checkpoint detected</span></div><div className="model-summary-grid"><div className="panel model-identity"><div className="model-visual"><BrainCircuit size={36}/><span className="visual-ring ring-one"/><span className="visual-ring ring-two"/></div><div className="eyebrow">MODEL ARCHITECTURE</div><h2>{model?.name || 'LRCN (CNN + BiLSTM)'}</h2><p>A convolutional feature extractor learns visual features per frame; a bidirectional LSTM models how those features change over time.</p><div className="tag-row"><span>PyTorch</span><span>OpenCV</span><span>Temporal video</span></div></div><div className="panel spec-panel"><PanelTitle title="Model configuration" subtitle="Loaded from the project checkpoint"/><div className="spec-list"><Spec label="Model checkpoint" value={model?.checkpoint || 'activity_model.pt'}/><Spec label="Activity classes" value={model?.class_count ?? 6}/><Spec label="Frames per clip" value={model?.sequence_length ?? 16}/><Spec label="Frame size" value={model?.image_size || '64 × 64'}/><Spec label="Inference device" value={model?.device || health?.device || 'CPU'}/><Spec label="API status" value={health?.status || 'unknown'} status={health?.status === 'ok'}/></div></div></div><div className="content-grid model-grid"><section className="panel"><PanelTitle title="Activity classes" subtitle="Labels supported by the loaded checkpoint"/>{(model?.classes || ['Walking', 'Running', 'Sitting', 'Standing', 'Lying_Down', 'Falling']).map((c, i) => <div className="class-row" key={c}><span className="class-color" style={{ background: COLORS[i % COLORS.length] }}/><div><strong>{pretty(c)}</strong><span>{c === 'Falling' ? 'Potential fall pattern' : 'Human movement class'}</span></div><span className="class-index">{String(i + 1).padStart(2, '0')}</span></div>)}</section><section className="panel"><PanelTitle title="Evaluation artifacts" subtitle="Files generated by evaluate.py"/><div className="artifact-row"><div className="artifact-icon"><BarChart3 size={18}/></div><div><strong>Classification report</strong><span>{metrics?.report?.startsWith('No evaluation') ? 'Not available' : 'outputs/classification_report.txt'}</span></div><span className={`artifact-status ${metrics?.report?.startsWith('No evaluation') ? '' : 'available'}`}>{metrics?.report?.startsWith('No evaluation') ? 'Missing' : 'Found'}</span></div><div className="artifact-row"><div className="artifact-icon"><BarChart3 size={18}/></div><div><strong>Confusion matrix</strong><span>outputs/confusion_matrix.png</span></div><span className={`artifact-status ${metrics?.confusion_matrix_available ? 'available' : ''}`}>{metrics?.confusion_matrix_available ? 'Found' : 'Missing'}</span></div><div className="report-box"><pre>{metrics?.report || 'Loading evaluation report…'}</pre></div></section></div><div className="alert alert-warning"><AlertTriangle size={18}/><div><strong>Important: validate the dataset split</strong><p>{metrics?.warning || 'Reported scores should not be interpreted as real-world accuracy without an independent test set.'} Audit file hashes and video sources before publishing accuracy claims.</p></div></div></>}
      </div>
      <footer className="footer"><span>© {new Date().getFullYear()} MotionAI</span><span><span className="status-dot online"/> Local AI workspace</span><span>Built with React · FastAPI · PyTorch</span></footer>
    </main>
    {toast && <div className="toast"><CheckCircle2 size={17}/>{toast}<button onClick={() => setToast('')}><X size={14}/></button></div>}
  </div>
}

const tooltipStyle = { background: '#151c2e', border: '1px solid #303951', borderRadius: 10, color: '#f2f5ff', fontSize: 12 }
function Metric({ icon: Icon, label, value, sub, accent }) { return <div className="metric-card"><div className="metric-top"><span className={`metric-icon ${accent}`}><Icon size={19}/></span><span className="metric-more">•••</span></div><div className="metric-value">{value}</div><div className="metric-label">{label}</div><div className="metric-sub">{sub}</div></div> }
function PanelTitle({ title, subtitle, action }) { return <div className="panel-title"><div><h3>{title}</h3><p>{subtitle}</p></div>{action}</div> }
function EmptyState({ icon: Icon, title, text, action, actionText }) { return <div className="empty-state"><div className="empty-icon"><Icon size={23}/></div><strong>{title}</strong><p>{text}</p>{action && <button className="text-action" onClick={action}>{actionText} <ArrowUpRight size={14}/></button>}</div> }
function HistoryTable({ rows = [], compact = false, onEmpty }) { if (!rows.length) return <EmptyState icon={Clock3} title="No activity recorded yet" text="Once you analyze a video, its prediction will appear here." action={onEmpty} actionText="Analyze your first video"/>; return <div className="table-scroll"><table className="data-table"><thead><tr><th>Activity</th><th>Source</th><th>Confidence</th><th>Status</th><th>Timestamp</th></tr></thead><tbody>{rows.map((r, i) => <tr key={r.id ?? `${r.timestamp}-${i}`}><td><div className="activity-cell"><span className={`activity-dot ${String(r.activity).toLowerCase().includes('fall') ? 'fall' : ''}`}/><strong>{pretty(r.activity)}</strong></div></td><td className="source-cell">{r.source || '—'}</td><td><div className="confidence-cell"><span>{Number(r.confidence || 0).toFixed(1)}%</span><div className="mini-track"><i style={{ width: `${Math.min(100, Number(r.confidence) || 0)}%` }}/></div></div></td><td>{r.is_fall ? <span className="status-badge danger"><AlertTriangle size={12}/> Fall alert</span> : <span className="status-badge neutral"><CheckCircle2 size={12}/> Classified</span>}</td><td className="time-cell">{dateLabel(r.timestamp)}</td></tr>)}</tbody></table></div> }
function Spec({ label, value, status }) { return <div className="spec-row"><span>{label}</span><strong>{status !== undefined && <i className={`status-dot ${status ? 'online' : ''}`}/>} {String(value)}</strong></div> }

createRoot(document.getElementById('root')).render(<React.StrictMode><App/></React.StrictMode>)
