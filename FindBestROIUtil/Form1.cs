using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System.Diagnostics;
using Emgu.CV.Features2D;
using Emgu.CV.GPU;
using Emgu.CV.CvEnum;
using Emgu.CV.UI;
using System.Threading;


namespace SlidingWindowTest
{
    public partial class Form1 : Form
    {
        Rectangle roi = Rectangle.Empty;
        Size RoiSize = new Size(300, 300);
        Image<Bgr, byte> image = null;
        Image<Gray, byte> grayimage = null;
        Image<Gray, Byte> model = null;
        Thread surfThread = null;
        Tuple<Rectangle, double> bestRoi = new Tuple<Rectangle, double>(Rectangle.Empty, 0.0f);

        public Form1()
        {
            InitializeComponent();

            model = new Image<Gray, Byte>("resources\\cereal_reese_puffs_1102.jpg");

            imageBox1.FunctionalMode = Emgu.CV.UI.ImageBox.FunctionalModeOption.Minimum;
            image = new Image<Bgr, Byte>("resources\\Wegmans-144.jpg").Resize(0.25,INTER.CV_INTER_CUBIC);
            grayimage = new Image<Gray, Byte>("resources\\Wegmans-144.jpg").Resize(0.25, INTER.CV_INTER_CUBIC);
            imageBox1.Image = image;

            MouseEventHandler ImageBoxClickHandler = new MouseEventHandler(ImageBoxClick);
            imageBox1.MouseDown += (s, a) => { ImageBoxClick(s, a); imageBox1.MouseMove += ImageBoxClickHandler; };
            imageBox1.MouseUp += delegate { imageBox1.MouseMove -= ImageBoxClickHandler; };
            this.MouseLeave += delegate { imageBox1.MouseMove -= ImageBoxClickHandler; };
        }

        public static Rectangle findBestRoi(Image<Gray, Byte> SceneImage, Image<Gray, Byte> ModelImage, Size RoiSize)
        {
            Point currentLoc = new Point(0, 0);
            Tuple<Rectangle, double> bestRoi = new Tuple<Rectangle, double>(Rectangle.Empty, 0.0f);

            Size StepSize = new Size((int)((float)RoiSize.Width / 2f), (int)((float)RoiSize.Height / 2f));

            while (currentLoc.Y < SceneImage.Height)
            {
                while(currentLoc.X < SceneImage.Width)
                {
                    double confidence;
                    long time;
                    PointF[] border;
                    Rectangle ThisROI = new Rectangle(currentLoc, RoiSize);
                    SceneImage.ROI = ThisROI;
                    SURFFeatureExample.DrawMatches.Draw(ModelImage, SceneImage, out time, out border, out confidence);
                    if (confidence > bestRoi.Item2) bestRoi = new Tuple<Rectangle, double>(ThisROI, confidence);
                    currentLoc.X += StepSize.Width;
                }
                currentLoc.X = 0;
                currentLoc.Y += StepSize.Height;
            }

            return bestRoi.Item1;
        }

        void ImageBoxClick(object sender, MouseEventArgs args)
        {
            int newX = args.Location.X - RoiSize.Width / 2;
            if (newX < 0) newX = 0;
            if (newX + RoiSize.Width > image.Width) newX = image.Width - RoiSize.Width;

            int newY = args.Location.Y - RoiSize.Height / 2;
            if (newY < 0) newY = 0;
            if (newY + RoiSize.Height > image.Height) newY = image.Height - RoiSize.Height;

            roi = new Rectangle(newX, newY, RoiSize.Width, RoiSize.Height);
            Image<Bgr, Byte> tmp = image.Clone();
            if (bestRoi.Item1 != Rectangle.Empty) tmp.Draw(bestRoi.Item1, new Bgr(0, 255, 0), 2);
            tmp.Draw(roi, new Bgr(0, 255, 255), 2);
            imageBox1.Image = tmp;


            if (surfThread != null)
            {
                surfThread.Abort();
            }

            surfThread = new Thread(() =>
            {
                double confidence;
                long time;
                PointF[] border;
                Image<Gray, Byte> tomatch = grayimage.Clone();
                tomatch.ROI = roi;
                SURFFeatureExample.DrawMatches.Draw(model, tomatch, out time, out border, out confidence);
                Image<Bgr, Byte> newImage = image.Clone();

                if (confidence > bestRoi.Item2)
                {
                    bestRoi = new Tuple<Rectangle, double>(roi, confidence);
                }

                if (border != null)
                {
                    for (int i = 0; i < border.Length; i++)
                    {
                        border[i].X += (float)roi.X;
                        border[i].Y += (float)roi.Y;
                    }
                    newImage.DrawPolyline(Array.ConvertAll<PointF, Point>(border, Point.Round), true, new Bgr(Color.Red), 5);
                }

                if (bestRoi.Item1 != Rectangle.Empty)
                {
                    newImage.Draw(bestRoi.Item1, new Bgr(0, 255, 0), 2);
                }

                newImage.Draw(roi, new Bgr(0, 255, 255), 2);

                imageBox1.Image = newImage;
            });
            surfThread.Start();

        }
    }
}

namespace SURFFeatureExample
{
    public static class DrawMatches
    {
        private static float ComputeScore(Matrix<float> Distances, Matrix<byte> Mask)
        {
            float score = 0.0f;
            int matches = 0;
            int ipoints = Distances.Rows;

            for (int i = 0; i < ipoints; i++)
            {
                if (Mask[i, 0] != 0)
                {
                    matches++;
                    score += Distances[i, 0];
                }
                else
                {
                    // score += 1;
                }
            }
            float final_score = 1f - (((float)ipoints - score) / (float)ipoints);
            //float final_score = (((float)ipoints - score) / (float)ipoints);
            //return 1f - (score * ((float)matches / (float)ipoints));
            //return score * ((float)matches/float(ipoints));
            //return ((score / (float)ipoints) * (((float)matches) / (float)ipoints));
            // return ((score / (float)matches) * (((float)matches) / (float)ipoints));
            return final_score;
        }

        /// <summary>
        /// Draw the model image and observed image, the matched features and homography projection.
        /// </summary>
        /// <param name="modelImage">The model image</param>
        /// <param name="observedImage">The observed image</param>
        /// <param name="matchTime">The output total time for computing the homography matrix.</param>
        /// <returns>The model image and observed image, the matched features and homography projection.</returns>
        public static Image<Bgr, Byte> Draw(Image<Gray, Byte> modelImage, Image<Gray, byte> observedImage, out long matchTime,
                                               out PointF[] border, out double confidence)
        {
            Stopwatch watch;
            HomographyMatrix homography = null;

            SURFDetector surfCPU = new SURFDetector(500, false);
            VectorOfKeyPoint modelKeyPoints;
            VectorOfKeyPoint observedKeyPoints;
            Matrix<int> indices;

            Matrix<byte> mask;
            int k = 2;
            double uniquenessThreshold = 0.8;
            if (GpuInvoke.HasCuda)
            {
                GpuSURFDetector surfGPU = new GpuSURFDetector(surfCPU.SURFParams, 0.01f);
                using (GpuImage<Gray, Byte> gpuModelImage = new GpuImage<Gray, byte>(modelImage))
                //extract features from the object image
                using (GpuMat<float> gpuModelKeyPoints = surfGPU.DetectKeyPointsRaw(gpuModelImage, null))
                using (GpuMat<float> gpuModelDescriptors = surfGPU.ComputeDescriptorsRaw(gpuModelImage, null, gpuModelKeyPoints))
                using (GpuBruteForceMatcher<float> matcher = new GpuBruteForceMatcher<float>(DistanceType.L2))
                {
                    modelKeyPoints = new VectorOfKeyPoint();
                    surfGPU.DownloadKeypoints(gpuModelKeyPoints, modelKeyPoints);
                    watch = Stopwatch.StartNew();

                    // extract features from the observed image
                    using (GpuImage<Gray, Byte> gpuObservedImage = new GpuImage<Gray, byte>(observedImage))
                    using (GpuMat<float> gpuObservedKeyPoints = surfGPU.DetectKeyPointsRaw(gpuObservedImage, null))
                    using (GpuMat<float> gpuObservedDescriptors = surfGPU.ComputeDescriptorsRaw(gpuObservedImage, null, gpuObservedKeyPoints))
                    using (GpuMat<int> gpuMatchIndices = new GpuMat<int>(gpuObservedDescriptors.Size.Height, k, 1, true))
                    using (GpuMat<float> gpuMatchDist = new GpuMat<float>(gpuObservedDescriptors.Size.Height, k, 1, true))
                    using (GpuMat<Byte> gpuMask = new GpuMat<byte>(gpuMatchIndices.Size.Height, 1, 1))
                    using (Stream stream = new Stream())
                    {
                        matcher.KnnMatchSingle(gpuObservedDescriptors, gpuModelDescriptors, gpuMatchIndices, gpuMatchDist, k, null, stream);
                        indices = new Matrix<int>(gpuMatchIndices.Size);
                        mask = new Matrix<byte>(gpuMask.Size);

                        //gpu implementation of voteForUniquess
                        using (GpuMat<float> col0 = gpuMatchDist.Col(0))
                        using (GpuMat<float> col1 = gpuMatchDist.Col(1))
                        {
                            GpuInvoke.Multiply(col1, new MCvScalar(uniquenessThreshold), col1, stream);
                            GpuInvoke.Compare(col0, col1, gpuMask, CMP_TYPE.CV_CMP_LE, stream);
                        }

                        observedKeyPoints = new VectorOfKeyPoint();
                        surfGPU.DownloadKeypoints(gpuObservedKeyPoints, observedKeyPoints);

                        //wait for the stream to complete its tasks
                        //We can perform some other CPU intesive stuffs here while we are waiting for the stream to complete.
                        stream.WaitForCompletion();

                        gpuMask.Download(mask);
                        gpuMatchIndices.Download(indices);

                        if (GpuInvoke.CountNonZero(gpuMask) >= 4)
                        {
                            int nonZeroCount = Features2DToolbox.VoteForSizeAndOrientation(modelKeyPoints, observedKeyPoints, indices, mask, 1.5, 20);
                            if (nonZeroCount >= 4)
                                homography = Features2DToolbox.GetHomographyMatrixFromMatchedFeatures(modelKeyPoints, observedKeyPoints, indices, mask, 2);
                        }

                        confidence = ComputeScore(gpuMatchDist.ToMatrix(), gpuMask.ToMatrix());

                        watch.Stop();
                    }
                }
            }
            else
            {
                //extract features from the object image
                modelKeyPoints = surfCPU.DetectKeyPointsRaw(modelImage, null);
                Matrix<float> modelDescriptors = surfCPU.ComputeDescriptorsRaw(modelImage, null, modelKeyPoints);

                watch = Stopwatch.StartNew();

                // extract features from the observed image
                observedKeyPoints = surfCPU.DetectKeyPointsRaw(observedImage, null);
                Matrix<float> observedDescriptors = surfCPU.ComputeDescriptorsRaw(observedImage, null, observedKeyPoints);
                BruteForceMatcher<float> matcher = new BruteForceMatcher<float>(DistanceType.L2);
                matcher.Add(modelDescriptors);

                indices = new Matrix<int>(observedDescriptors.Rows, k);
                using (Matrix<float> dist = new Matrix<float>(observedDescriptors.Rows, k))
                {
                    matcher.KnnMatch(observedDescriptors, indices, dist, k, null);
                    mask = new Matrix<byte>(dist.Rows, 1);
                    mask.SetValue(255);
                    Features2DToolbox.VoteForUniqueness(dist, uniquenessThreshold, mask);
                    confidence = ComputeScore(dist, mask);
                }

                int nonZeroCount = CvInvoke.cvCountNonZero(mask);
                if (nonZeroCount >= 4)
                {
                    nonZeroCount = Features2DToolbox.VoteForSizeAndOrientation(modelKeyPoints, observedKeyPoints, indices, mask, 1.5, 20);
                    if (nonZeroCount >= 4)
                        homography = Features2DToolbox.GetHomographyMatrixFromMatchedFeatures(modelKeyPoints, observedKeyPoints, indices, mask, 2);
                }

                watch.Stop();
            }

            //Draw the matched keypoints
            Image<Bgr, Byte> result = Features2DToolbox.DrawMatches(modelImage, modelKeyPoints, observedImage, observedKeyPoints,
               indices, new Bgr(255, 255, 255), new Bgr(255, 255, 255), mask, Features2DToolbox.KeypointDrawType.DEFAULT);

            #region draw the projected region on the image
            if (homography != null)
            {  //draw a rectangle along the projected model
                Rectangle rect = modelImage.ROI;
                PointF[] pts = new PointF[] {
                    new PointF(rect.Left, rect.Bottom),
                    new PointF(rect.Right, rect.Bottom),
                    new PointF(rect.Right, rect.Top),
                    new PointF(rect.Left, rect.Top)};
                homography.ProjectPoints(pts);
                border = pts;

                result.DrawPolyline(Array.ConvertAll<PointF, Point>(pts, Point.Round), true, new Bgr(Color.Red), 5);
            }
            else
            {
                border = null;
            }
            #endregion

            matchTime = watch.ElapsedMilliseconds;

            return result;
        }
    }
}