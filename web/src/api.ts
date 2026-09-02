export interface Diagnostic {
  code: string;
  message: string;
  source_key: string | null;
  fields: string[];
}

export interface Provenance {
  producer: string | null;
  detector: string | null;
  checkpoint: string | null;
  tracker: string | null;
  post_processing: string | null;
  adaptation_iterations: number | null;
  notes: string | null;
}

export interface SourceMetadata {
  source_key: string;
  sequence: string;
  frame_numbering: "one_based";
  frame_count: number;
  width: number;
  height: number;
  frame_rate: number;
  source_hash: string;
  adapter: "mot_gt_9" | "mot_result_10";
  source_class: "ground_truth" | "tracker_result";
  policy_classification:
    | "ground_truth_training_source"
    | "local_test_adapted_development_material"
    | "local_development_material";
  source_row_count: number;
  observation_count: number;
  capability: {
    id_status: "tracked" | "sentinel_only" | "unusable";
    track_features: boolean;
    usable_track_ids: number[];
    diagnostics: Diagnostic[];
  };
  provenance: Provenance;
  diagnostics: Diagnostic[];
}

export interface SequenceListResponse {
  sources: SourceMetadata[];
  unavailable: Array<{
    source_key: string;
    sequence: string;
    diagnostic: Diagnostic;
  }>;
  diagnostics: Diagnostic[];
}

export interface Observation {
  source_key: string;
  sequence: string;
  frame: number;
  row_index: number;
  row_hash: string;
  source_hash: string;
  raw_track_id: number;
  usable_track_id: number | null;
  raw_geometry: { x: number; y: number; width: number; height: number };
  display_geometry: { x1: number; y1: number; x2: number; y2: number };
  score: number | null;
  ground_truth: { mark: number; class_id: number; visibility: number } | null;
  opaque_result_fields: [number, number, number] | null;
  score_semantics: "tracker_score" | "not_defined";
  ground_truth_semantics: "mot_mark_class_visibility" | "not_defined";
}

export interface FrameObservationsResponse {
  source_key: string;
  sequence: string;
  frame: number;
  frame_numbering: "one_based";
  source_hash: string;
  observations: Observation[];
}

export interface CropResponse {
  source_key: string;
  sequence: string;
  source_hash: string;
  frame: number;
  row_index: number;
  row_hash: string;
  media_type: "image/jpeg";
  image_base64: string;
}

export interface TrackGap {
  start_frame: number;
  end_frame: number;
  length: number;
}

export interface TrackEvidenceResponse {
  source_key: string;
  sequence: string;
  source_hash: string;
  track_id: number;
  observation_frames: number[];
  gaps: TrackGap[];
  first_observation: Observation;
  last_observation: Observation;
  previous_observation: Observation | null;
  next_observation: Observation | null;
  observations: Observation[];
}

export interface FilmstripSample {
  is_current: boolean;
  observation: Observation;
}

export interface FilmstripResponse {
  source_key: string;
  sequence: string;
  source_hash: string;
  track_id: number;
  current_row_index: number;
  total_observations: number;
  sampled_count: number;
  samples: FilmstripSample[];
}

export interface ContextPairEvidence {
  frame: number;
  focal_row_index: number;
  competitor_row_index: number;
  focal_raw_xywh: [number, number, number, number];
  competitor_raw_xywh: [number, number, number, number];
  iou: number;
  edge_distance_pixels: number;
  focal_box_height: number;
  normalized_edge_proximity: number;
}

export interface ContextCompetitor {
  rank: number;
  track_id: number;
  best_iou: number;
  best_normalized_edge_proximity: number;
  comparison_count: number;
  evidence: ContextPairEvidence[];
}

export interface TrackContextResponse {
  source_key: string;
  sequence: string;
  source_hash: string;
  track_id: number;
  geometry_basis: "raw_xywh";
  window: {
    center_frame: number;
    start_frame: number;
    end_frame: number;
    radius: number;
  };
  requested_count: number;
  hard_cap: number;
  total_competitors: number;
  competitors: ContextCompetitor[];
}

export interface EventSettings {
  displacement_enabled: boolean;
  displacement_threshold: number;
  displacement_operator: "greater_than_or_equal";
  scale_change_enabled: boolean;
  scale_change_threshold: number;
  scale_change_operator: "greater_than_or_equal";
  close_interaction_enabled: boolean;
  close_interaction_threshold: number;
  close_interaction_operator: "less_than_or_equal";
}

export interface TimelineEventsResponse {
  source_key: string;
  sequence: string;
  source_hash: string;
  track_id: number;
  geometry_basis: string;
  settings: EventSettings;
  confidence: {
    status: "meaningful" | "absent" | "constant" | "sentinel";
    meaningful: boolean;
    score_semantics: "tracker_score" | "not_defined";
    threshold: number;
    threshold_operator: "less_than_or_equal";
    diagnostic: Diagnostic | null;
  };
  displacement_events: Array<{ from_frame: number; to_frame: number; threshold: number }>;
  scale_change_events: Array<{ from_frame: number; to_frame: number; threshold: number }>;
  close_interaction_events: Array<{ frame: number; threshold: number }>;
  low_confidence_observations: Observation[];
}

export class ApiRequestError extends Error {
  constructor(public readonly status: number, message: string) {
    super(message);
  }
}

export async function fetchSequences(signal?: AbortSignal): Promise<SequenceListResponse> {
  const response = await fetch("/api/sequences", { signal });
  if (!response.ok) {
    throw new Error(`Source metadata request failed (${response.status})`);
  }
  return (await response.json()) as SequenceListResponse;
}

export function frameImageUrl(source: SourceMetadata, frame: number): string {
  const key = encodeURIComponent(source.source_key);
  const hash = encodeURIComponent(source.source_hash);
  return `/api/sequences/${key}/frames/${frame}?source_hash=${hash}`;
}

export async function fetchFrameObservations(
  source: SourceMetadata,
  frame: number,
  signal?: AbortSignal,
): Promise<FrameObservationsResponse> {
  const key = encodeURIComponent(source.source_key);
  const hash = encodeURIComponent(source.source_hash);
  const response = await fetch(
    `/api/sequences/${key}/frames/${frame}/observations?source_hash=${hash}`,
    { signal },
  );
  if (!response.ok) {
    throw new Error(`Frame observation request failed (${response.status})`);
  }
  return (await response.json()) as FrameObservationsResponse;
}

export async function fetchObservationCrop(
  source: SourceMetadata,
  rowIndex: number,
  signal?: AbortSignal,
): Promise<CropResponse> {
  const key = encodeURIComponent(source.source_key);
  const hash = encodeURIComponent(source.source_hash);
  const response = await fetch(
    `/api/sequences/${key}/observations/${rowIndex}/crop?source_hash=${hash}&padding=16&max_size=256`,
    { signal },
  );
  if (!response.ok) {
    throw new Error(`Observation crop request failed (${response.status})`);
  }
  return (await response.json()) as CropResponse;
}

export async function fetchTrackSearch(
  source: SourceMetadata,
  trackId: number,
  signal?: AbortSignal,
): Promise<TrackEvidenceResponse> {
  const key = encodeURIComponent(source.source_key);
  const search = new URLSearchParams({
    track_id: String(trackId),
    source_hash: source.source_hash,
  });
  const response = await fetch(`/api/sequences/${key}/tracks?${search}`, { signal });
  if (!response.ok) {
    throw new ApiRequestError(response.status, `Track search request failed (${response.status})`);
  }
  return (await response.json()) as TrackEvidenceResponse;
}

export async function fetchTrackFilmstrip(
  source: SourceMetadata,
  trackId: number,
  currentRowIndex: number,
  signal?: AbortSignal,
): Promise<FilmstripResponse> {
  const key = encodeURIComponent(source.source_key);
  const search = new URLSearchParams({
    current_row_index: String(currentRowIndex),
    source_hash: source.source_hash,
  });
  const response = await fetch(
    `/api/sequences/${key}/tracks/${trackId}/filmstrip?${search}`,
    { signal },
  );
  if (!response.ok) {
    throw new Error(`Filmstrip request failed (${response.status})`);
  }
  const filmstrip = (await response.json()) as FilmstripResponse;
  if (filmstrip.sampled_count > 64 || filmstrip.samples.length > 64) {
    throw new Error("Filmstrip response exceeded 64 samples");
  }
  return filmstrip;
}

export async function fetchTrackEvidence(
  source: SourceMetadata,
  trackId: number,
  currentRowIndex: number,
  signal?: AbortSignal,
): Promise<TrackEvidenceResponse> {
  const key = encodeURIComponent(source.source_key);
  const search = new URLSearchParams({
    current_row_index: String(currentRowIndex),
    source_hash: source.source_hash,
  });
  const response = await fetch(`/api/sequences/${key}/tracks/${trackId}?${search}`, { signal });
  if (!response.ok) {
    throw new ApiRequestError(response.status, `Track evidence request failed (${response.status})`);
  }
  return (await response.json()) as TrackEvidenceResponse;
}

export async function fetchTrackContext(
  source: SourceMetadata,
  trackId: number,
  frame: number,
  count: number,
  signal?: AbortSignal,
): Promise<TrackContextResponse> {
  const key = encodeURIComponent(source.source_key);
  const search = new URLSearchParams({
    frame: String(frame),
    count: String(Math.min(8, Math.max(0, Math.trunc(count)))),
    source_hash: source.source_hash,
  });
  const response = await fetch(`/api/sequences/${key}/tracks/${trackId}/context?${search}`, { signal });
  if (!response.ok) {
    throw new ApiRequestError(response.status, `Track context request failed (${response.status})`);
  }
  return (await response.json()) as TrackContextResponse;
}

export async function fetchTimelineEvents(
  source: SourceMetadata,
  trackId: number,
  settings?: EventSettings,
  signal?: AbortSignal,
): Promise<TimelineEventsResponse> {
  const key = encodeURIComponent(source.source_key);
  const search = new URLSearchParams({ source_hash: source.source_hash });
  if (settings !== undefined) {
    search.set("enable_displacement", String(settings.displacement_enabled));
    search.set("displacement_threshold", String(settings.displacement_threshold));
    search.set("enable_scale_change", String(settings.scale_change_enabled));
    search.set("scale_change_threshold", String(settings.scale_change_threshold));
    search.set("enable_close_interaction", String(settings.close_interaction_enabled));
    search.set("close_interaction_threshold", String(settings.close_interaction_threshold));
  }
  const response = await fetch(`/api/sequences/${key}/tracks/${trackId}/events?${search}`, { signal });
  if (!response.ok) {
    throw new ApiRequestError(response.status, `Timeline event request failed (${response.status})`);
  }
  return (await response.json()) as TimelineEventsResponse;
}