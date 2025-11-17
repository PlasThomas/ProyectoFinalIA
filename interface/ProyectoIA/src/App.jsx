import { useState, useRef, useEffect, useCallback } from 'react';
import './App.css'; 

// URL base dinámica que funciona con proxy de Vite
const getApiBaseUrl = () => {
  const hostname = window.location.hostname;
  
  // Si estamos en ngrok, usa el path relativo que será proxyado por Vite
  if (hostname.includes('ngrok-free.dev')) {
    return '/api'; // Esto será redirigido a localhost:8000 por el proxy de Vite
  }
  
  // Para desarrollo local
  return 'http://localhost:8000';
};

const API_BASE_URL = getApiBaseUrl();

console.log('🔧 API Base URL configurada:', API_BASE_URL);
console.log('🌐 Host actual:', window.location.hostname);

function App() {
  // Estado para manejar la emoción detectada (inicialmente null)
  const [emotion, setEmotion] = useState(null);
  // Estado para almacenar todas las probabilidades (para mostrar más detalles)
  const [probabilities, setProbabilities] = useState({});
  // Estado que manejará si el escaneo está en progreso
  const [isScanning, setIsScanning] = useState(false);
  // Referencias para el video (muestra la cámara) y el canvas (toma la foto)
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  // Estado para manejar el stream de la cámara y los errores
  const [stream, setStream] = useState(null);
  const [error, setError] = useState(null);
  
  // Función para iniciar la cámara
  const startCamera = useCallback(async () => {
    setError(null);
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setError("Error de compatibilidad: Tu navegador no soporta la API de la cámara.");
      return;
    }
    
    try {
      // Solicitar acceso a la cámara frontal (ideal para selfies)
      const newStream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          facingMode: "user", // Usa la cámara frontal en móvil
          width: { ideal: 400 },
          height: { ideal: 300 }
        } 
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = newStream;
        // Reproducir el video tan pronto como sea posible
        videoRef.current.play().catch(e => console.error("Error al reproducir el video:", e));
        setStream(newStream); // Establecer el stream DESPUÉS de asignarlo al video
      }
    } catch (err) {
      // MANEJO DE ERRORES MÁS ESPECÍFICO
      console.error("Error detallado al acceder a la cámara:", err);
      
      if (err.name === 'NotAllowedError' || err.name === 'SecurityError') {
        setError("ACCESO DENEGADO: Debes otorgar permisos de cámara al navegador.");
      } else if (err.name === 'NotFoundError') {
        setError("CÁMARA NO ENCONTRADA: Asegúrate de tener una cámara activa.");
      } else {
        setError(`Error desconocido al iniciar la cámara: ${err.name} (${err.message})`);
      }
    }
  }, []); // Dependencias vacías para useCallback

  // Efecto que inicia la cámara al cargar el componente y la detiene al desmontar
  useEffect(() => {
    startCamera();

    // Función de limpieza para detener el stream de la cámara
    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        const currentStream = videoRef.current.srcObject;
        currentStream.getTracks().forEach(track => track.stop());
      }
    };
  }, [startCamera]); // Solo depende de startCamera

  // Función principal para capturar el frame y enviarlo al backend
  const handleScan = async () => {
    // Asegurarse de que el video esté cargado, el canvas exista y no estemos escaneando
    if (!videoRef.current || !canvasRef.current || isScanning || !videoRef.current.srcObject) return;

    setIsScanning(true);
    setEmotion(null);
    setProbabilities({});
    setError(null);

    try {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      
      // 1. Capturar la imagen del video al canvas
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      // Dibujar la imagen invertida para mantener el efecto espejo de la cámara frontal
      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      ctx.setTransform(1, 0, 0, 1, 0, 0); // Resetear la transformación

      // 2. Convertir el canvas a un Blob (archivo)
      const imageBlob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.9));
      
      if (!imageBlob) {
        throw new Error("Fallo al crear la imagen para el escaneo.");
      }

      // 3. Preparar la solicitud FormData, igual que lo hiciste con curl
      const formData = new FormData();
      formData.append('image', imageBlob, 'face_capture.jpg'); 
      formData.append('detect_face', 'true');

      console.log('📤 Enviando solicitud a:', `${API_BASE_URL}/predict`);

      // 4. Enviar la imagen al endpoint /predict de FastAPI
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        body: formData,
        // Agregar headers para mejor compatibilidad
        headers: {
          'Accept': 'application/json',
        },
      });
      
      // Verificar si la respuesta es OK antes de intentar parsear JSON
      if (!response.ok) {
        let errorDetail;
        try {
          const errorData = await response.json();
          errorDetail = errorData.detail || `Error ${response.status}`;
        } catch {
          errorDetail = `Error ${response.status}: ${response.statusText}`;
        }
        throw new Error(errorDetail);
      }
      
      const data = await response.json();
      
      // 5. Mostrar los resultados
      setEmotion(data.label);
      setProbabilities(data.probabilities);

    } catch (err) {
      console.error("❌ Fallo durante el escaneo:", err);
      setError(err.message || "Error desconocido al comunicarse con el servidor de IA.");
    } finally {
      setIsScanning(false);
    }
  };

  // Función para ordenar las probabilidades de mayor a menor
  const sortedProbabilities = Object.entries(probabilities)
    .sort(([, a], [, b]) => b - a)
    .filter(([, prob]) => prob > 0.01); // Filtra las que son irrelevantes

  return (
    <div className="facial-scanner-app">
      {/* Título y descripción */}
      <header className="app-header">
        <h1>👁️ Identificador de Emociones IA</h1>
        <p className="subtitle">Escanea un rostro para detectar su estado emocional.</p>
        {/* Información de debug */}
        <div style={{ fontSize: '12px', color: '#666', marginTop: '10px' }}>
          <div>🔗 Conectando a: {API_BASE_URL}</div>
          <div>🌐 Host: {window.location.hostname}</div>
        </div>
      </header>

      {/* Área de Visualización de la Cámara */}
      <div className="camera-preview-container">
        {/* Muestra un mensaje de error si la cámara falla */}
        {error && <p className="error-message">{error}</p>}
        
        {/* Elemento de Video: Muestra el stream de la cámara (oculto si hay error) */}
        <video 
          ref={videoRef} 
          className="camera-video"
          autoPlay 
          playsInline 
          muted 
          // Solo muestra el video si hay un stream activo
          style={{ display: videoRef.current?.srcObject && !error ? 'block' : 'none' }}
        />
        
        {/* Elemento Canvas: Oculto, solo se usa para tomar la foto (frame) */}
        <canvas 
          ref={canvasRef} 
          style={{ display: 'none' }} 
        />
        
        {/* Placeholder mientras la cámara carga o si no hay stream y no hay error */}
        {(!videoRef.current?.srcObject && !error) && (
            <div className="camera-placeholder">
              <div className="spinner"></div>
              <p>Esperando acceso a la cámara...</p>
            </div>
        )}
      </div>
      
      {/* Sección de Resultados */}
      <section className="results-area">
        <h2>Resultado del Análisis</h2>
        <div className="emotion-display">
          {emotion ? (
            <>
              <p className="detected-emotion">
                Emoción detectada: <strong>{emotion}</strong>
              </p>
              {/* Mostrar todas las probabilidades */}
              <div className="probabilities-list">
                <h3>Otras probabilidades:</h3>
                {sortedProbabilities.map(([label, prob]) => (
                  <p key={label} className="probability-item">
                    {label}: {(prob * 100).toFixed(1)}%
                  </p>
                ))}
              </div>
            </>
          ) : (
            <p className="pending-message">
              {isScanning 
                ? 'Procesando en el servidor de IA...' 
                : (videoRef.current?.srcObject ? 'Presiona "Escanear Rostro" para analizar.' : 'Esperando cámara...')}
            </p>
          )}
        </div>
      </section>

      {/* Botón de Acción */}
      <footer className="action-footer">
        <button 
          onClick={handleScan} 
          // Desactivar si está escaneando, no hay stream o hay un error
          disabled={isScanning || !videoRef.current?.srcObject || error}
          className="scan-button"
        >
          {isScanning ? 'Procesando...' : '📸 Escanear Rostro'}
        </button>
      </footer>
    </div>
  );
}

export default App;