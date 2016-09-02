using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using Emgu.CV;
using Emgu.CV.Structure;
using System.Drawing;

namespace FindBestROIUtil.Tests
{
    [TestClass()]
    public class FindBestROIUtilTests
    {
        [TestMethod()]
        public void FindBestROITest()
        {
            Image<Gray, Byte> Scene = new Image<Gray, Byte>("resources\\Wegmans-144.jpg");
            Image<Gray, Byte> Model = new Image<Gray, Byte>("resources\\cereal_reese_puffs_1102.jpg");
            Rectangle BestROI = FindBestROIUtil.FindBestROI(Scene, Model, new Size(800, 800));
            Assert.AreEqual(BestROI, new Rectangle(400, 1200, 800, 800));
        }
    }
}