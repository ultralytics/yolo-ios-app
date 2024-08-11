//  Ultralytics YOLO 🚀 - AGPL-3.0 License
//
//  HumanModel for Ultralytics YOLO App
// This struct is designed to turn the inference results of the YOLOv8-Human model into a manageable DataModel of human feature values ​​in the Ultralytics YOLO app. When in tracking mode, this struct averages the feature values ​​of a given individual across frames to a stable value.
// This struct automatically analyzes the boxes, scores, and feature values ​​provided to the update function to create a human model.//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app


import Foundation
import UIKit

let updateFrequency: Int = 120

struct Person {
    var index: Int
    var box: CGRect = .zero
    
    var score: Float = 0
    var weight: Float = 0
    var height: Float = 0
    
    var age: Int = 0
    
    var gender: String = "female"
    var genderConfidence: Float = 0
    var race: String = "asian"
    var raceConfidence: Float = 0
    
    var listCount: Int = 0
    var scoreRawList: [Float] = []
    var weightRawList: [Float] = []
    var heightRawList: [Float] = []
    var ageRawList: [Float] = []
    var maleRawList: [Float] = []
    var femaleRawList: [Float] = []
    var asianRawList: [Float] = []
    var whiteRawList: [Float] = []
    var middleEasternRawList: [Float] = []
    var indianRawList: [Float] = []
    var latinoRawList: [Float] = []
    var blackRawList: [Float] = []

    var trackedBox: CGRect?
    var color:UIColor
    
    var unDetectedCounter: Int = 0
    var stable = false
    
    init(index: Int) {
        self.index = index
        self.color = UIColor(red: CGFloat.random(in: 0...1),
                     green: CGFloat.random(in: 0...1),
                     blue: CGFloat.random(in: 0...1),
                     alpha: 0.6)
    }
    
    mutating func update(box:CGRect, score:Float, features:[Float]) {
        self.box = box
        if scoreRawList.count >= updateFrequency {
            scoreRawList.removeFirst()
            weightRawList.removeFirst()
            heightRawList.removeFirst()
            ageRawList.removeFirst()
            maleRawList.removeFirst()
            femaleRawList.removeFirst()
            asianRawList.removeFirst()
            whiteRawList.removeFirst()
            middleEasternRawList.removeFirst()
            indianRawList.removeFirst()
            latinoRawList.removeFirst()
            blackRawList.removeFirst()
        }
        

        self.scoreRawList.append(score)
        self.weightRawList.append(Float(applyPolyModel(weight: Double(features[0]))))
        self.heightRawList.append(features[1])
        self.ageRawList.append(features[2])
        self.femaleRawList.append(features[3])
        self.maleRawList.append(features[4])
        self.asianRawList.append(features[5])
        self.whiteRawList.append(features[6])
        self.middleEasternRawList.append(features[7])
        self.indianRawList.append(features[8])
        self.latinoRawList.append(features[9])
        self.blackRawList.append(features[10])
        calcurateFeatures()
                
        self.unDetectedCounter = 0
    }
    
    private mutating func calcurateFeatures() {

        self.score = average(of: scoreRawList)
        self.weight = average(of: weightRawList)
        self.height = average(of: heightRawList)
        self.age = Int(round(average(of: ageRawList)))
        let femaleAverage = average(of: femaleRawList)
        let maleAverage = average(of: maleRawList)
        let genderCandidates = [femaleAverage,maleAverage]
        var genderMaxIndex = 0
        var genderMaxValue = genderCandidates[0]

        for (genderIndex, genderValue) in genderCandidates.dropFirst().enumerated() {
            if genderValue > genderMaxValue {
                genderMaxValue = genderValue
                genderMaxIndex = genderIndex + 1
            }
        }
        
        self.gender = genders[genderMaxIndex]
        self.genderConfidence = genderMaxValue
        
        let asianAverage = average(of: asianRawList)
        let whiteAverage = average(of: whiteRawList)
        let middleEasternAverage = average(of: middleEasternRawList)
        let indianAverage = average(of: indianRawList)
        let latinoAverage = average(of: latinoRawList)
        let blackAverage = average(of: blackRawList)

        let raceCandidates =  [asianAverage,whiteAverage,middleEasternAverage,indianAverage,latinoAverage,blackAverage]
        var raceMaxIndex = 0
        var raceMaxValue = raceCandidates[0]

        for (raceIndex, raceValue) in raceCandidates.dropFirst().enumerated() {
            if raceValue > raceMaxValue {
                raceMaxValue = raceValue
                raceMaxIndex = raceIndex + 1
            }
        }
        self.race = races[raceMaxIndex]
        self.raceConfidence = raceMaxValue
    }
    
    func average(of numbers: [Float]) -> Float {
        guard !numbers.isEmpty else {
            return 0
        }
        var sum: Float = 0
        for number in numbers {
            sum += number
        }
        return sum / Float(numbers.count)
    }

    private mutating func applyPolyModel(weight: Double, degree: Int = 4) -> Double {
        let coef: [Double] = [0, 44.196, -1.2491, 0.014767, -6.2096e-05]
        let intercept: Double = -477.7766285997004

        let weight2 = weight * weight
        let weight3 = weight2 * weight
        let weight4 = weight3 * weight

        let result = coef[0]
                    + coef[1] * weight
                    + coef[2] * weight2
                    + coef[3] * weight3
                    + coef[4] * weight4
                    + intercept

        return result
    }
}

let genders = ["female", "male"]
let races = ["asian", "white", "middle eastern", "indian", "latino", "black"]

