import React, { useState, useEffect, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import "./App.css";

function App() {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState("");
    const [model, setModel] = useState(null);
    const [words, setWords] = useState([]);
    const [classes, setClasses] = useState([]);
    const [intents, setIntents] = useState([]);
    const messagesEndRef = useRef(null);

    // Cargar recursos al iniciar
    useEffect(() => {
        async function loadResources() {
            try {
                // Cargar el modelo
                const loadedModel = await tf.loadLayersModel("/model/model.json");
                setModel(loadedModel);

                // Cargar palabras, clases e intents
                const wordsResponse = await fetch("words.json");
                setWords(await wordsResponse.json());

                const classesResponse = await fetch("classes.json");
                setClasses(await classesResponse.json());

                const intentsResponse = await fetch("intents.json");
                setIntents(await intentsResponse.json());

                console.log("Recursos cargados");
            } catch (error) {
                console.error("Error al cargar los recursos:", error);
            }
        }

        loadResources();
    }, []);

    // Tokenizar y procesar la entrada del usuario
    const tokenize = (sentence) => {
        return sentence.toLowerCase().match(/\b(\w+)\b/g);
    };

    const bagOfWords = (sentence, words) => {
        const sentenceTokens = tokenize(sentence);
        const bag = Array(words.length).fill(0);
        sentenceTokens.forEach((token) => {
            const index = words.indexOf(token);
            if (index !== -1) {
                bag[index] = 1;
            }
        });
        return bag;
    };

    // Predecir la clase
    const predictClass = async (sentence) => {
        if (!model) {
            return null; // Si el modelo no está listo, no predecir nada
        }

        const inputBag = bagOfWords(sentence, words);
        const inputTensor = tf.tensor([inputBag]);

        const predictions = await model.predict(inputTensor).array();
        const maxIdx = predictions[0].indexOf(Math.max(...predictions[0]));

        return predictions[0][maxIdx] > 0.25 ? classes[maxIdx] : null;
    };

    // Obtener una respuesta según la clase
    const getResponse = (intentTag) => {
        const intent = intents.intents?.find((i) => i.tag === intentTag);
        return intent
            ? intent.responses[
                  Math.floor(Math.random() * intent.responses.length)
              ]
            : "No entiendo lo que dices.";
    };

    // Manejar el envío del mensaje
    const handleSend = async () => {
        const trimmedInput = input.trim(); // Validar entrada
        if (trimmedInput === "") return;

        const newMessage = { text: trimmedInput, sender: "user" };
        setMessages((prevMessages) => [...prevMessages, newMessage]);
        setInput(""); // Limpiar el input

        // Obtener la predicción del chatbot
        if (model) {
            const intent = await predictClass(trimmedInput);
            const response = getResponse(intent);

            const botMessage = { text: response, sender: "bot" };
            setMessages((prevMessages) => [...prevMessages, botMessage]);
        } else {
            const botMessage = {
                text: "Estoy cargando el modelo. Por favor, inténtalo de nuevo en unos segundos.",
                sender: "bot",
            };
            setMessages((prevMessages) => [...prevMessages, botMessage]);
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
