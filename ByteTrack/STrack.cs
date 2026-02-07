using Microsoft.Win32.SafeHandles;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace ByteTrack
{
    // Equivalente al enum TrackState en C++
    public enum TrackState
    {
        New = 0,
        Tracked,
        Lost,
        Removed
    }

    public class STrack
    {
        public bool is_activated;
        public int track_id;
        public TrackState state;

        public List<float> _tlwh;
        public List<float> tlwh;
        public List<float> tlbr;

        public float x2;
        public float y2;

        public int frame_id;
        public int tracklet_len;
        public int start_frame;

        public KAL_MEAN mean;
        public KAL_COVA covariance;
        public float score;

        private static int _count = 0;
        private KalmanFilter kalman_filter;

        private const float Eps = 1e-6f;

        // Constructor
        public STrack(List<float> tlwh_, float score, float x2, float y2)
        {
            _tlwh = new List<float>(tlwh_);
            is_activated = false;
            track_id = 0;
            state = TrackState.New;

            tlwh = new List<float>() { 0, 0, 0, 0 };
            tlbr = new List<float>() { 0, 0, 0, 0 };

            static_tlwh();
            static_tlbr();

            frame_id = 0;
            tracklet_len = 0;
            this.score = score;
            start_frame = 0;
            this.x2 = x2;
            this.y2 = y2;   
        }

        public void activate(KalmanFilter kalman_filter, int frame_id)
        {
            this.kalman_filter = kalman_filter;
            this.track_id = this.next_id();

            // Copiamos _tlwh en un array temporal
            List<float> _tlwh_tmp = new List<float>(_tlwh);

            // Convertimos a xyah
            List<float> xyah = tlwh_to_xyah(_tlwh_tmp);

            DETECTBOX xyah_box = new DETECTBOX();
            for (int i = 0; i < 4; i++)
            {
                xyah_box[i] = xyah[i];
            }

            // Iniciamos Kalman
            var mc = this.kalman_filter.initiate(xyah_box);
            this.mean = mc.Item1;
            this.covariance = mc.Item2;

            static_tlwh();
            static_tlbr();

            this.tracklet_len = 0;
            this.state = TrackState.Tracked;
            if (frame_id == 1)
            {
                this.is_activated = true;
            }

            this.frame_id = frame_id;
            this.start_frame = frame_id;
        }

        public void re_activate(STrack new_track, int frame_id, bool new_id = false)
        {
            // Convertimos la tlwh de new_track a xyah
            List<float> xyah = tlwh_to_xyah(new_track.tlwh);

            DETECTBOX xyah_box = new DETECTBOX();
            for (int i = 0; i < 4; i++)
            {
                xyah_box[i] = xyah[i];
            }

            // Kalman update
            var mc = this.kalman_filter.update(this.mean, this.covariance, xyah_box);
            this.mean = mc.Item1;
            this.covariance = mc.Item2;

            static_tlwh();
            static_tlbr();

            this.tracklet_len = 0;
            this.state = TrackState.Tracked;
            this.is_activated = true;
            this.frame_id = frame_id;
            this.score = new_track.score;

            if (new_id)
                this.track_id = next_id();
        }

        public void update(STrack new_track, int frame_id)
        {
            this.frame_id = frame_id;
            this.tracklet_len++;

            // Convertimos a xyah
            List<float> xyah = tlwh_to_xyah(new_track.tlwh);

            DETECTBOX xyah_box = new DETECTBOX();
            for (int i = 0; i < 4; i++)
            {
                xyah_box[i] = xyah[i];
            }

            var mc = this.kalman_filter.update(this.mean, this.covariance, xyah_box);
            this.mean = mc.Item1;
            this.covariance = mc.Item2;

            static_tlwh();
            static_tlbr();

            this.state = TrackState.Tracked;
            this.is_activated = true;

            this.score = new_track.score;
        }

        private void static_tlwh()
        {
            if (this.state == TrackState.New)
            {
                // Usa directamente _tlwh
                for (int i = 0; i < 4; i++)
                {
                    tlwh[i] = _tlwh[i];
                }
                return;
            }

            // mean => [ x_center, y_center, ratio, h, ... ]
            tlwh[0] = mean[0];
            tlwh[1] = mean[1];
            tlwh[2] = mean[2];
            tlwh[3] = mean[3];

            // asegurar altura positiva mínima
            if (tlwh[3] < Eps) tlwh[3] = Eps;

            // ancho = ratio * h
            tlwh[2] *= tlwh[3];

            // pasamos de (center_x, center_y) a topleft
            tlwh[0] -= tlwh[2] / 2f;
            tlwh[1] -= tlwh[3] / 2f;
        }

        private void static_tlbr()
        {
            tlbr.Clear();
            tlbr.AddRange(tlwh);
            // tlbr[2] += tlbr[0]
            tlbr[2] = tlbr[2] + tlbr[0];
            // tlbr[3] += tlbr[1]
            tlbr[3] = tlbr[3] + tlbr[1];
        }

        public List<float> tlwh_to_xyah(List<float> tlwh_tmp)
        {
            // tlwh => [x, y, w, h]
            // => xyah => [x_center, y_center, w/h, h]
            List<float> output = new List<float>(tlwh_tmp);
            // asegurar h > 0 para evitar división por cero
            if (output[3] < Eps) output[3] = Eps;
            // x_center
            output[0] += output[2] / 2f;
            // y_center
            output[1] += output[3] / 2f;
            // ratio = w / h (clamp h)
            output[2] = output[2] / output[3];
            return output;
        }

        public List<float> to_xyah()
        {
            return tlwh_to_xyah(tlwh);
        }

        public static List<float> tlbr_to_tlwh(ref List<float> tlbr)
        {
            // [x1, y1, x2, y2] => [x1, y1, w, h]
            tlbr[2] = tlbr[2] - tlbr[0];
            tlbr[3] = tlbr[3] - tlbr[1];
            return tlbr;
        }

        public void mark_lost()
        {
            state = TrackState.Lost;
        }

        public void mark_removed()
        {
            state = TrackState.Removed;
        }

        private int next_id()
        {
            _count++;
            return _count;
        }

        public int end_frame()
        {
            return this.frame_id;
        }

        public static void multi_predict(List<STrack> stracks, KalmanFilter kalman_filter)
        {
            for (int i = 0; i < stracks.Count; i++)
            {
                if (stracks[i].state != TrackState.Tracked)
                {
                    // anular velocidad de la altura si no está en estado Tracked
                    stracks[i].mean[7] = 0;
                }
                kalman_filter.predict(stracks[i].mean, stracks[i].covariance);
                stracks[i].static_tlwh();
                stracks[i].static_tlbr();
            }
        }
    }
}
