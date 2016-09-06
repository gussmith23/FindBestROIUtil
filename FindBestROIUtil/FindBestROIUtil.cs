using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Features2D;
using Emgu.CV.GPU;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System;
using System.Diagnostics;
using System.Drawing;

namespace FindBestROIUtil
{
    public static class FindBestROIUtil
    {
        /// <summary>
        /// Scans scene image patch-by-patch and returns ROI with the highest-confidence SURF match with
        /// the model image.
        /// Each patch is of size PatchSize. Starting at the top left of the image, a sliding box pans
        /// left-to-right, moving half of the PatchSize width each time. When it hits the edge, it 
        /// goes back to zero and moves down half of PatchSize height.
        /// The smaller the ROI, the longer this algorithm will take to run.
        /// </summary>
        /// <param name="SceneImage">The image to search within.</param>
        /// <param name="ModelImage">The model image to search for.</param>
        /// <param name="PatchSize">The size of the patch to slide across the image.</param>
        /// <returns>The patch with the most confident SURF match to the model image.</returns>
        public static Rectangle FindBestROI(Image<Gray, Byte> SceneImage, Image<Gray, Byte> ModelImage, Size PatchSize, out PointF CenterRelativeToPatch)
        {
            Point CurrentLoc = new Point(0, 0);
            Tuple<Rectangle, double> BestROI = new Tuple<Rectangle, double>(Rectangle.Empty, 0.0f);
            Size StepSize = new Size((int)((float)PatchSize.Width / 2f), (int)((float)PatchSize.Height / 2f));
            Size SceneSize = new Size(SceneImage.Width, SceneImage.Height);

            CenterRelativeToPatch = PointF.Empty;

            while (CurrentLoc.Y + PatchSize.Height < SceneSize.Height)
            {
                if (CurrentLoc.Y + PatchSize.Height > SceneSize.Height)
                    CurrentLoc.Y = SceneSize.Height - PatchSize.Height;

                while (CurrentLoc.X + PatchSize.Width < SceneSize.Width)
                {
                    if (CurrentLoc.X + PatchSize.Width > SceneSize.Width)
                        CurrentLoc.X = SceneSize.Width - PatchSize.Width;

                    double confidence;
                    long time;
                    PointF[] border;
                    Rectangle ThisROI = new Rectangle(CurrentLoc, PatchSize);
                    SceneImage.ROI = ThisROI;
                    Draw(ModelImage, SceneImage, out time, out border, out confidence);
                    if (confidence > BestROI.Item2)
                    {
                        BestROI = new Tuple<Rectangle, double>(ThisROI, confidence);

                        #region calculate center of object
                        if (border != null)
                        {
                            float TotalX = 0, TotalY = 0;
                            for (int i = 0; i < 4; i++)
                            {
                                TotalX += 0.25f * border[i].X;
                                TotalY += 0.25f * border[i].Y;
                            }
                            CenterRelativeToPatch = new PointF(TotalX, TotalY);
                        }
                        #endregion
                    }
                    CurrentLoc.X += StepSize.Width;
                }
                CurrentLoc.X = 0;
                CurrentLoc.Y += StepSize.Height;
            }

            return BestROI.Item1;
        }

        /// <summary>
        /// Return a confidence score for a set of match distances.
        /// </summary>
        /// <param name="Distances">The match distances.</param>
        /// <param name="Mask"></param>
        /// <returns></returns>
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
            return final_score;
        }

        /// <summary>
        /// Draw the model image and observed image, the matched features and homography projection.
        /// Based on EMGU's SURF example.
        /// </summary>
        /// <param name="modelImage">The model image</param>
        /// <param name="observedImage">The observed image</param>
        /// <param name="matchTime">The output total time for computing the homography matrix.</param>
        /// <returns>The model image and observed image, the matched features and homography projection.</returns>
        private static Image<Bgr, Byte> Draw(Image<Gray, Byte> modelImage, Image<Gray, byte> observedImage, out long matchTime,
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
