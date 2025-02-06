import React, { useRef, useEffect } from 'react';
import WaveSurfer from 'wavesurfer.js';

const Waveform = ({ src, currentTime, setCurrentTime, isPlaying, duration }) => {
    const waveformRef = useRef(null);
    const wavesurferRef = useRef(null);

    useEffect(() => {
        wavesurferRef.current = WaveSurfer.create({
            container: waveformRef.current,
            waveColor: '#ddd',
            progressColor: '#333',
            cursorColor: '#333',
            height: 200,
            barWidth: 2,
            responsive: true,
        });

        wavesurferRef.current.load(src);

        wavesurferRef.current.on('audioprocess', () => {
            setCurrentTime(wavesurferRef.current.getCurrentTime());
        });

        wavesurferRef.current.on('seek', (progress) => {
            setCurrentTime(progress * wavesurferRef.current.getDuration());
        });

        return () => wavesurferRef.current.destroy();
    }, [src, setCurrentTime]);

    useEffect(() => {
        if (isPlaying) {
            wavesurferRef.current.play();
        } else {
            wavesurferRef.current.pause();
        }
    }, [isPlaying]);

    return <div ref={waveformRef} className="waveform"></div>;
};

export default Waveform;