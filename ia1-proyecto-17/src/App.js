import React, { useState, useEffect, useRef } from "react";
import "./App.css";

function App() {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState("");
    const messagesEndRef = useRef(null);

    const handleSend = () => {
        const trimmedInput = input.trim(); // Validar entrada
        if (trimmedInput !== "") {
            const newMessage = { text: trimmedInput, sender: "user" };
            setMessages((prevMessages) => [...prevMessages, newMessage]);
            setInput(""); // Limpiar el input

            // Generar respuesta automática del chatbot
            setTimeout(() => {
                const botResponses = [
                    "¡Hola! ¿Cómo puedo ayudarte?",
                    "¿Podrías darme más detalles?",
                    "¡Gracias por tu mensaje! Estoy aquí para ayudarte.",
                ];
                const botMessage = {
                    text: botResponses[Math.floor(Math.random() * botResponses.length)],
                    sender: "bot",
                };
                setMessages((prevMessages) => [...prevMessages, botMessage]);
            }, 1000); // Respuesta después de 1 segundo
        }
    };

    // Auto-scroll hacia el último mensaje
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages]);

    return (
        <div className="chatbot-container">
            <header className="chatbot-header">Chatbot</header>
            <div className="chatbot-messages" aria-live="polite">
                {messages.map((msg, index) => (
                    <div
                        key={index}
                        className={`message ${msg.sender === "user" ? "user" : "bot"}`}
                    >
                        {msg.text}
                    </div>
                ))}
                <div ref={messagesEndRef} />
            </div>
            <div className="chatbot-input">
                <input
                    type="text"
                    placeholder="Escribe tu mensaje..."
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    aria-label="Escribe tu mensaje"
                />
                <button
                    onClick={handleSend}
                    aria-label="Enviar mensaje"
                    disabled={input.trim() === ""}
                >
                    Enviar
                </button>
            </div>
        </div>
    );
}

export default App;
