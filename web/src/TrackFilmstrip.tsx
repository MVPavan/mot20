import { useEffect, useState } from "react";

import { fetchObservationCrop, type FilmstripResponse, type Observation, type SourceMetadata } from "./api";

function FilmstripCrop({ observation, source }: { observation: Observation; source: SourceMetadata }) {
  const [dataUrl, setDataUrl] = useState<string | null>(null);
  const [error, setError] = useState(false);
  useEffect(() => {
    const controller = new AbortController();
    let active = true;
    setDataUrl(null);
    setError(false);
    fetchObservationCrop(source, observation.row_index, controller.signal)
      .then((crop) => {
        if (!active) return;
        if (
          crop.source_key !== source.source_key ||
          crop.source_hash !== source.source_hash ||
          crop.frame !== observation.frame ||
          crop.row_index !== observation.row_index ||
          crop.row_hash !== observation.row_hash
        ) {
          throw new Error("Crop response identity did not match the filmstrip observation");
        }
        setDataUrl(`data:${crop.media_type};base64,${crop.image_base64}`);
      })
      .catch((requestError: unknown) => {
        if (active && !(requestError instanceof DOMException && requestError.name === "AbortError")) setError(true);
      });
    return () => { active = false; controller.abort(); };
  }, [observation, source]);
  if (error) return <span className="filmstrip__crop-state">Crop unavailable</span>;
  if (dataUrl === null) return <span className="filmstrip__crop-state">Loading crop</span>;
  return <img alt="" src={dataUrl} />;
}

export function TrackFilmstrip({ filmstrip, onSeek, source }: { filmstrip: FilmstripResponse; onSeek(frame: number): void; source: SourceMetadata }) {
  return (
    <section className="filmstrip" aria-label={`Track filmstrip, ${filmstrip.sampled_count} samples`}>
      <div className="filmstrip__list">
        {filmstrip.samples.map((sample, index) => {
          const position = sample.is_current ? "Current" : index === 0 ? "First" : index === filmstrip.samples.length - 1 ? "Last" : sample.observation.frame < filmstrip.samples.find((item) => item.is_current)!.observation.frame ? "Earlier" : "Later";
          return (
            <button aria-label={`${position} crop, seek frame ${sample.observation.frame}`} key={sample.observation.row_index} onClick={() => onSeek(sample.observation.frame)} type="button">
              <FilmstripCrop observation={sample.observation} source={source} />
              <span>{position} / {sample.observation.frame}</span>
            </button>
          );
        })}
      </div>
    </section>
  );
}