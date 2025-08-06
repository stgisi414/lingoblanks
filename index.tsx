import React, { useState, useEffect, useRef } from 'react';
import ReactDOM from 'react-dom/client';
import { GoogleGenAI, Type } from "@google/genai";

const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

// --- Interfaces ---
interface Lesson {
  title: string;
  languageCode: string; // e.g., 'es-ES' for Spanish
  article: string; // The article with placeholders like '{{word}}'
  targetWords: string[];
  nextTopicSuggestion: string;
}

// --- WAV Utility ---
/**
 * Creates a WAV file buffer from raw PCM data. The Gemini TTS API returns raw
 * 16-bit PCM audio at a 24000Hz sample rate, which needs a proper WAV header
 * for the browser's Web Audio API to decode.
 * @param {Uint8Array} pcmData The raw PCM data.
 * @param {number} sampleRate The sample rate (e.g., 24000).
 * @param {number} numChannels The number of channels (e.g., 1).
 * @param {number} bitsPerSample The number of bits per sample (e.g., 16).
 * @returns {ArrayBuffer} The complete WAV file as an ArrayBuffer.
 */
function pcmToWav(pcmData: Uint8Array, sampleRate: number, numChannels: number, bitsPerSample: number): ArrayBuffer {
    const header = new ArrayBuffer(44);
    const view = new DataView(header);
    
    const dataSize = pcmData.length;
    const blockAlign = (numChannels * bitsPerSample) / 8;
    const byteRate = sampleRate * blockAlign;

    const writeString = (view: DataView, offset: number, string: string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };

    // RIFF chunk descriptor
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + dataSize, true); // file-size - 8
    writeString(view, 8, 'WAVE');
    
    // "fmt " sub-chunk
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true); // chunk size
    view.setUint16(20, 1, true); // audio format (1 = PCM)
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitsPerSample, true);

    // "data" sub-chunk
    writeString(view, 36, 'data');
    view.setUint32(40, dataSize, true);
    
    const wavBytes = new Uint8Array(44 + pcmData.length);
    wavBytes.set(new Uint8Array(header), 0);
    wavBytes.set(pcmData, 44);
    
    return wavBytes.buffer;
}

// --- Utility function to shuffle an array (Fisher-Yates algorithm) ---
function shuffleArray<T>(array: T[]): T[] {
    const shuffled = [...array];
    for (let i = shuffled.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    return shuffled;
}

// --- App Component ---
const App = () => {
    // State management
    const [language, setLanguage] = useState(() => {
        return localStorage.getItem('lastLanguage') || 'Spanish';
    });
    const [topic, setTopic] = useState(() => {
        return localStorage.getItem('lastTopic') || 'ordering food at a restaurant';
    });
    const [difficulty, setDifficulty] = useState(() => {
        return localStorage.getItem('lastDifficulty') || 'normal';
     });
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [lesson, setLesson] = useState<Lesson | null>(null);
    const [lessonImage, setLessonImage] = useState<string | null>(null);
    
    const [userAnswers, setUserAnswers] = useState<string[]>([]);
    const [isCorrect, setIsCorrect] = useState<boolean[]>([]);
    const [showResults, setShowResults] = useState(false);
    
    const [shuffledWords, setShuffledWords] = useState<string[]>([]);

    const [isAudioLoading, setIsAudioLoading] = useState(false);
    const [isAudioPlaying, setIsAudioPlaying] = useState(false);
    
    const [nextTopicCooldown, setNextTopicCooldown] = useState(0);

    // Refs for audio playback
    const audioContextRef = useRef<AudioContext | null>(null);
    const audioSourceRef = useRef<AudioBufferSourceNode | null>(null);

    // --- Effects ---
    // Cleanup audio context on unmount
    useEffect(() => {
        return () => {
            audioContextRef.current?.close();
            if (audioSourceRef.current) {
                try {
                   audioSourceRef.current.stop();
                   audioSourceRef.current.disconnect();
                } catch(e) {
                   console.warn("Could not stop audio source", e)
                }
            }
        };
    }, []);

    // Cooldown timer for the next topic button
    useEffect(() => {
        if (nextTopicCooldown > 0) {
            const timer = setTimeout(() => setNextTopicCooldown(nextTopicCooldown - 1), 1000);
            return () => clearTimeout(timer);
        }
    }, [nextTopicCooldown]);
    
    // Save language and topic to local storage whenever they change
    useEffect(() => {
        localStorage.setItem('lastLanguage', language);
    }, [language]);
    
    // Save topic to local storage
    useEffect(() => {
        localStorage.setItem('lastTopic', topic);
    }, [topic]);

    // Save difficulty to local storage
    useEffect(() => {
        localStorage.setItem('lastDifficulty', difficulty);
    }, [difficulty]);

    // --- Gemini API Calls ---
    const generateLesson = async (currentTopic: string) => {
        const difficultyMap: Record<string, string> = {
            'easy': 'Use short, simple sentences with common vocabulary and basic grammar.',
            'normal': 'Use standard sentences with a mix of common and some less common vocabulary.',
            'hard': 'Use longer, more complex sentences with advanced vocabulary and grammar.',
        };

        setIsLoading(true);
        setError(null);
        setLesson(null);
        setLessonImage(null);
        setShowResults(false);

        try {
            // 1. Generate Lesson Content
            const lessonPrompt = `Create a language lesson for a beginner learning ${language} on the topic of "${currentTopic}". The difficulty is ${difficulty}. ${difficultyMap[difficulty]} Provide a title, the IETF language code for ${language}, a short article (3-4 sentences) with exactly 5 words replaced by "{{word}}" for a fill-in-the-blank exercise, a JSON array of those 5 exact string words for the "targetWords" field (the words should be the actual words, not placeholders), and a suggestion for a follow-up lesson topic.`;
            
            const response = await ai.models.generateContent({
                model: "gemini-2.5-flash",
                contents: lessonPrompt,
                config: {
                    responseMimeType: "application/json",
                    responseSchema: {
                        type: Type.OBJECT,
                        properties: {
                            title: { type: Type.STRING },
                            languageCode: { type: Type.STRING },
                            article: { type: Type.STRING },
                            targetWords: { type: Type.ARRAY, items: { type: Type.STRING } },
                            nextTopicSuggestion: { type: Type.STRING },
                        }
                    }
                }
            });

            const lessonData = JSON.parse(response.text) as Lesson;
            
            if (!lessonData.targetWords || lessonData.targetWords.length === 0) {
              throw new Error("API returned invalid lesson data (missing target words).");
            }
            
            setLesson(lessonData);
            setUserAnswers(new Array(lessonData.targetWords.length).fill(''));
            setIsCorrect([]);
            setShuffledWords(shuffleArray(lessonData.targetWords));

            // 2. Generate Header Image
            const imagePrompt = `A vibrant, minimalist, educational illustration for a language lesson titled "${lessonData.title}".`;
            const imageResponse = await ai.models.generateImages({
                model: 'imagen-3.0-generate-002',
                prompt: imagePrompt,
                config: {
                    numberOfImages: 1,
                    outputMimeType: 'image/jpeg',
                    aspectRatio: '16:9',
                },
            });
            
            const base64ImageBytes = imageResponse.generatedImages[0].image.imageBytes;
            const imageUrl = `data:image/jpeg;base64,${base64ImageBytes}`;
            setLessonImage(imageUrl);

        } catch (err: any) {
            console.error("Failed to generate lesson:", err);
            setError(`An error occurred: ${err.message}. Please try again.`);
        } finally {
            setIsLoading(false);
            setNextTopicCooldown(5); // Add a 5-second cooldown
        }
    };

    const handleListen = async (textToListen: string) => {
        if (isAudioLoading || isAudioPlaying || !lesson) return;

        setIsAudioLoading(true);
        setError(null);

        try {
            if (audioSourceRef.current) {
                audioSourceRef.current.stop();
            }

            const response = await ai.models.generateContent({
                model: "gemini-2.5-flash-preview-tts", // FIX: Use the correct TTS model
                contents: [{ parts: [{ text: textToListen }] }],
                config: {
                    responseModalities: ['AUDIO'],
                    speechConfig: {
                        languageCode: lesson.languageCode,
                        voiceConfig: {
                            prebuiltVoiceConfig: { voiceName: 'Kore' },
                        },
                    },
                },
            });
            
            const audioDataB64 = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
            if (!audioDataB64) {
                throw new Error("No audio data returned from API.");
            }

            const pcmData = Uint8Array.from(atob(audioDataB64), c => c.charCodeAt(0));
            const wavBuffer = pcmToWav(pcmData, 24000, 1, 16);

            if (!audioContextRef.current || audioContextRef.current.state === 'closed') {
                audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
            }
            const audioContext = audioContextRef.current;

            const audioBuffer = await audioContext.decodeAudioData(wavBuffer);
            const source = audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(audioContext.destination);
            source.start();

            audioSourceRef.current = source;
            setIsAudioPlaying(true);

            source.onended = () => {
                setIsAudioPlaying(false);
                audioSourceRef.current = null;
            };

        } catch (err: any) {
            console.error("Failed to play audio.", err);
            setError(`Failed to play audio: ${err.message}`);
        } finally {
            setIsAudioLoading(false);
        }
    };
    
    // --- Event Handlers ---
    const handleAnswerChange = (index: number, value: string) => {
        if (showResults) return;
        const newAnswers = [...userAnswers];
        newAnswers[index] = value;
        setUserAnswers(newAnswers);
    };

    const handleCheckAnswers = () => {
        if (!lesson) return;
        const results = userAnswers.map((answer, index) =>
            answer.trim().toLowerCase() === lesson.targetWords[index].trim().toLowerCase()
        );
        setIsCorrect(results);
        setShowResults(true);
    };

    const handleNextTopic = () => {
        if (lesson?.nextTopicSuggestion && nextTopicCooldown === 0) {
            setTopic(lesson.nextTopicSuggestion);
            generateLesson(lesson.nextTopicSuggestion);
        }
    };
    
    const renderArticle = () => {
        if (!lesson) return null;
        // Regex to split by {{...}} placeholders, keeping the delimiters.
        // This robustly handles cases where the API returns {{word}} or {{actual word}}.
        const parts = lesson.article.split(/(\{\{[^}]+\}\}|\{\{word\}\})/g);
        let inputIndex = 0;

        return (
            <p className="article">
                {parts.map((part, i) => {
                    if (part.startsWith('{{') && part.endsWith('}}')) {
                        const currentIndex = inputIndex;
                        inputIndex++;
                        return (
                            <input
                                key={i}
                                type="text"
                                className={`article-input ${showResults ? (isCorrect[currentIndex] ? 'correct' : 'incorrect') : ''}`}
                                value={userAnswers[currentIndex] || ''}
                                onChange={(e) => handleAnswerChange(currentIndex, e.target.value)}
                                disabled={showResults}
                                aria-label={`Blank word ${currentIndex + 1}`}
                            />
                        );
                    }
                    // Render the text part if it's not an empty string
                    return part ? <React.Fragment key={i}>{part}</React.Fragment> : null;
                })}
            </p>
        );
    };
    
    return (
        <div className="container">
            <header>
                <h1><img
                        src="/logo.png"
                        alt="LingoBlanks Logo"
                        style={{
                            width: '75px',
                            height: '75px',
                            verticalAlign: 'middle',
                            marginTop: '-7.5px'
                        }}
                    /> LingoBlanks</h1>
            </header>
            <main>
                <div className="controls">
                    <div className="control-group">
                        <label htmlFor="language-input">Language</label>
                        <input
                            id="language-input"
                            type="text"
                            value={language}
                            onChange={(e) => setLanguage(e.target.value)}
                            placeholder="e.g., French, Japanese"
                            disabled={isLoading}
                        />
                    </div>
                     <div className="control-group">
                        <label htmlFor="topic-input">Topic</label>
                        <input
                            id="topic-input"
                            type="text"
                            value={topic}
                            onChange={(e) => setTopic(e.target.value)}
                            placeholder="e.g., At the airport"
                            disabled={isLoading}
                        />
                    </div>
                    <div className="control-group">
                        <label>Difficulty</label>
                        <div style={{display: 'flex', gap: '1rem'}}>
                            <label>
                                <input
                                    type="radio"
                                    value="easy"
                                    checked={difficulty === 'easy'}
                                    onChange={(e) => setDifficulty(e.target.value)}
                                    disabled={isLoading}
                                />
                                Easy
                            </label>
                            <label>
                                <input
                                    type="radio"
                                    value="normal"
                                    checked={difficulty === 'normal'}
                                    onChange={(e) => setDifficulty(e.target.value)}
                                    disabled={isLoading}
                                />
                                Normal
                            </label>
                            <label>
                                <input
                                    type="radio"
                                    value="hard"
                                    checked={difficulty === 'hard'}
                                    onChange={(e) => setDifficulty(e.target.value)}
                                    disabled={isLoading}/>Hard</label>
                        </div>
                    </div>
                </div>

                <div className="main-cta">
                     <button onClick={() => generateLesson(topic)} disabled={isLoading || !language || !topic}>
                        {isLoading ? 'Generating...' : 'Start Lesson'}
                    </button>
                </div>
                
                {error && <p className="error-message">{error}</p>}
                
                {isLoading && !lesson && <p className="status-message">Generating your lesson...</p>}

                {lesson && (
                    <div className="lesson-content">
                        {lessonImage ? (
                            <img src={lessonImage} alt={lesson.title} className="lesson-header-image" />
                        ) : (
                            <div className="lesson-header-image" style={{ backgroundColor: '#eee' }}></div>
                        )}
                        <div style={{padding: '0 2rem 2rem'}}>
                            <h2 className="lesson-title">{lesson.title}</h2>

                            <div className="listen-controls">
                               <button onClick={() => handleListen(lesson.article.replace(/\{\{[^}]+\}\}/g, 'blank'))} disabled={isAudioLoading || isAudioPlaying}>
                                    {isAudioLoading ? 'Loading...' : isAudioPlaying ? 'Playing...' : 'Listen to Article'}
                                </button>
                            </div>

                            {renderArticle()}
                            
                            {!showResults && (
                                <div className="word-bank">
                                    <h3>Word Bank</h3>
                                    <div className="word-bank-words">
                                        {shuffledWords.map(word => (
                                           <button
                                               key={word}
                                               className="word-bank-word"
                                               onClick={() => handleListen(word)}
                                               disabled={isAudioLoading || isAudioPlaying}
                                           >{word}</button>
                                        ))}    
                                    </div>
                                </div>
                            )}

                            <div className="action-buttons">
                                {!showResults ? (
                                    <button onClick={handleCheckAnswers}>Check Answers</button>
                                ) : (
                                    <>
                                       <button onClick={handleNextTopic} disabled={nextTopicCooldown > 0} className="secondary-btn">
                                            Next Topic: {lesson.nextTopicSuggestion}
                                        </button>
                                        {nextTopicCooldown > 0 && <span className="countdown-timer">({nextTopicCooldown}s)</span>}
                                    </>
                                )}
                            </div>
                        </div>
                    </div>
                )}
            </main>
        </div>
    );
};

const root = ReactDOM.createRoot(document.getElementById('root')!);
root.render(<App />);