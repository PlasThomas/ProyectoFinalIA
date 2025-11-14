import { useState } from 'react';
import './App.css';

// Componente principal de la aplicación
function App() {
  // Estado para manejar la emoción detectada (inicialmente vacío)
  const [emotion, setEmotion] = useState(null);

  // Estado que manejará si el escaneo está en progreso
  const [isScanning, setIsScanning] = useState(false);

  // Función placeholder para manejar la acción de tomar la foto y escanear
  const handleScan = () => {
    setIsScanning(true);
    setEmotion(null); // Limpiar resultado anterior

    // **Aquí irá la lógica real:**
    // 1. Acceder a la cámara (usando navigator.mediaDevices.getUserMedia)
    // 2. Tomar la foto (Capturar el frame del video)
    // 3. Enviar la imagen al modelo de IA para su análisis
    
    // Simulación de un escaneo (reemplazar con la llamada a la IA real)
    setTimeout(() => {
      const detectedEmotion = "Felicidad"; // Resultado de la IA simulado
      setEmotion(detectedEmotion);
      setIsScanning(false);
    }, 3000); // Simulación de 3 segundos de procesamiento
  };

  return (
    <div className="facial-scanner-app">
      {/* Título y descripción */}
      <header className="app-header">
        <h1>👁️ Identificador de Emociones IA</h1>
        <p className="subtitle">Escanea un rostro para detectar su estado emocional.</p>
      </header>

      {/* Área de Visualización (Aquí irá el componente de la cámara) */}
      <div className="camera-preview-container">
        {/* Placeholder para la cámara. Aquí se integrará <video> o <canvas> */}
        <div className="camera-placeholder">
          {/*  */}
          {isScanning ? (
            <div className="scanning-message">
              <div className="spinner"></div>
              <p>Escaneando rostro...</p>
            </div>
          ) : (
            <p>Zona de previsualización de la cámara</p>
          )}
        </div>
      </div>
      
      {/* Sección de Resultados */}
      <section className="results-area">
        <h2>Resultado del Análisis</h2>
        <div className="emotion-display">
          {emotion ? (
            <p className="detected-emotion">
              Emoción detectada: <strong>{emotion}</strong>
            </p>
          ) : (
            <p className="pending-message">
              {isScanning ? 'Procesando...' : 'Presiona "Escanear Rostro" para comenzar.'}
            </p>
          )}
        </div>
      </section>

      {/* Botón de Acción */}
      <footer className="action-footer">
        <button 
          onClick={handleScan} 
          disabled={isScanning}
          className="scan-button"
        >
          {isScanning ? 'Procesando...' : '📸 Escanear Rostro'}
        </button>
      </footer>
    </div>
  );
}

export default App;