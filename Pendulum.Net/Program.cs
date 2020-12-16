using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NumSharp;
using SixLabors.ImageSharp;
using Ebby.Gym.Envs.Classic;
using System.Threading;
using SharpLearning.XGBoost.Learners;
using SharpLearning.InputOutput.Csv;
using System.IO;
using SharpLearning.Containers.Matrices;
using CsvHelper;
using CsvHelper.Configuration;
using System.Globalization;
using SharpLearning.FeatureTransformations;
using System.Reflection.Emit;
using SharpLearning.Containers.Extensions;
using System.Runtime.Remoting.Messaging;

namespace Pendulum.Net
{
    class Program
    {
        static void Main(string[] args)
        {
            //CartPoleEnv();

            List<Passenger> passengers = GetPassengerData(@"C:\Projekt\Pendulum\Dataset\Titanic\complete.csv");
            passengers.Shuffle(new Random());
            var trainPassengers = passengers.Take((int)(passengers.Count() * 0.8)).ToList();
            var testPassengers = passengers.Skip((int)(passengers.Count()*0.8)).Take((int)(passengers.Count() * 0.2)).ToList();

            var featuresTrain = GetFeatures(trainPassengers);
            var featuresTest = GetFeatures(testPassengers);

            //var featuresTest = GetFeatures(testPassengers);
            double[] labelsTrain = trainPassengers.Select(p => (double)p.survived).ToArray();
            double[] labelsTest = testPassengers.Select(p => (double)p.survived).ToArray();


            //F64Matrix observationsTest = new F64Matrix(featuresTest.SelectMany(i => i).ToArray(), featuresTest.Count(), featuresTest.First().Length);

            var learner = new RegressionXGBoostLearner(silent: false, maximumTreeDepth: 4, estimators:500, dropoutRate:0.1);
            using (SharpLearning.XGBoost.Models.RegressionXGBoostModel model = learner.Learn(featuresTrain, labelsTrain))
            {
                var predictions = model.Predict(featuresTrain);
                double accTrainData = GetAccuracy(predictions, labelsTrain);

                var predictionsTest = model.Predict(featuresTest);
                double accTestData = GetAccuracy(predictionsTest, labelsTest);


            }


        }

        private static F64Matrix GetFeatures(List<Passenger> trainPassengers)
        {
            var features = new List<double[]>();
            var sexOneHot = OneHotEncodeFeatures(trainPassengers.Select(p => p.sex));
            var parchOneHot = OneHotEncodeFeatures(trainPassengers.Select(p => p.parch));
            var cabinOneHot = OneHotEncodeFeatures(trainPassengers.Select(p => p.cabin));
            var embarkedOneHot = OneHotEncodeFeatures(trainPassengers.Select(p => p.embarked));
            for (int i = 0; i < trainPassengers.Count; i++)
            {
                Passenger p = trainPassengers[i];
                var dList = new List<double>();
                dList.Add(p.pclass);
                dList.Add(p.age ?? -1);
                dList.Add(p.sibsp);
                dList.Add(p.fare);
                dList.AddRange(sexOneHot[i]);
                dList.AddRange(parchOneHot[i]);

                dList.AddRange(cabinOneHot[i]);
                dList.AddRange(embarkedOneHot[i]);
                features.Add(dList.ToArray());
            }

            var matrix = new F64Matrix(features.SelectMany(i => i).ToArray(), features.Count(), features.First().Length);
            return matrix;
        }

        private static List<Passenger> GetPassengerData(string path)
        {
            TextReader reader = new StreamReader(path);
            var passengerList = new List<Passenger>();
            using (var csv = new CsvReader(reader, new CsvConfiguration(CultureInfo.InvariantCulture) { Delimiter = ",", HasHeaderRecord = true }))
            {

                int i = 0;
                int unidentified = 0;

                while (csv.Read())
                {
                    i++;
                    //var dyn = csv.GetRecord<dynamic>();
                    try
                    {
                        var passenger = csv.GetRecord<Passenger>();
                        passengerList.Add(passenger);
                    }
                    catch (Exception e)
                    {
                        unidentified++;
                    }
                }
            }
            return passengerList;
        }

        public static List<double[]> OneHotEncodeFeatures(IEnumerable<string> categories)
        {
            var list = new List<double[]>();
            IEnumerable<string> nrOfCategories = categories.Distinct();
            foreach (var category in categories)
            {
                var row = new List<double>();
                foreach (var distincCategory in nrOfCategories)
                {
                    row.Add(distincCategory == category ? 1 : 0);
                }
                list.Add(row.ToArray());
            }
            return list;
        }
        public static double GetAccuracy(double[] predictions, double[] truths)
        {
            var accList = new List<int>();
            for (int i = 0; i < predictions.Length; i++)
            {
                if (Math.Round(predictions[i]) == truths[i])
                    accList.Add(1);
                else
                    accList.Add(0);
            }
            return (double)accList.Sum() / accList.Count();
        }

    }
    public class Passenger
    {
        
        public int survived { get; set; }
        public double pclass { get; set; }
        public string name { get; set; }
        public string sex { get; set; }
        public double? age { get; set; }
        public int sibsp { get; set; }
        public string parch { get; set; }
        public string ticket { get; set; }
        public double fare { get; set; }
        public string cabin { get; set; }
        public string embarked { get; set; }

    }


}
