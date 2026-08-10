export interface VideoFrameSourceInfo {
  width: number;
  height: number;
  duration: number;
  frameCount: number;
  fps: number;
}

/** Loads a video file and lets you seek to and grab individual sampled frames from it. */
export class VideoFrameSource {
  readonly info: VideoFrameSourceInfo;
  private video: HTMLVideoElement;
  private objectUrl: string;

  private constructor(video: HTMLVideoElement, objectUrl: string, info: VideoFrameSourceInfo) {
    this.video = video;
    this.objectUrl = objectUrl;
    this.info = info;
  }

  static async load(file: File, fps: number): Promise<VideoFrameSource> {
    const objectUrl = URL.createObjectURL(file);
    const video = document.createElement('video');
    video.muted = true;
    video.playsInline = true;
    video.preload = 'auto';
    video.src = objectUrl;

    try {
      await new Promise<void>((resolve, reject) => {
        video.onloadedmetadata = () => resolve();
        video.onerror = () => reject(new Error('Could not load this video file.'));
      });
    } catch (err) {
      URL.revokeObjectURL(objectUrl);
      throw err;
    }

    const duration = video.duration;
    if (!Number.isFinite(duration) || duration <= 0) {
      URL.revokeObjectURL(objectUrl);
      throw new Error("Could not determine this video's duration.");
    }

    const frameCount = Math.max(1, Math.round(duration * fps));

    return new VideoFrameSource(video, objectUrl, {
      width: video.videoWidth,
      height: video.videoHeight,
      duration,
      frameCount,
      fps,
    });
  }

  /** Seeks to the given sampled frame index and returns the positioned video element to draw from. */
  async seekToFrame(index: number): Promise<HTMLVideoElement> {
    const targetTime = Math.min(index / this.info.fps, Math.max(0, this.info.duration - 1 / this.info.fps));
    await this.seekTo(targetTime);
    return this.video;
  }

  private seekTo(time: number): Promise<void> {
    return new Promise((resolve, reject) => {
      const video = this.video;
      if (Math.abs(video.currentTime - time) < 1e-4) {
        resolve();
        return;
      }

      const cleanup = () => {
        video.removeEventListener('seeked', onSeeked);
        video.removeEventListener('error', onError);
      };
      const onSeeked = () => {
        cleanup();
        resolve();
      };
      const onError = () => {
        cleanup();
        reject(new Error('Failed to seek within this video.'));
      };

      video.addEventListener('seeked', onSeeked);
      video.addEventListener('error', onError);
      video.currentTime = time;
    });
  }

  dispose(): void {
    this.video.pause();
    this.video.removeAttribute('src');
    this.video.load();
    URL.revokeObjectURL(this.objectUrl);
  }
}
