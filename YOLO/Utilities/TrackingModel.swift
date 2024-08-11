//  Ultralytics YOLO 🚀 - AGPL-3.0 License
//
//  HumanModel for Ultralytics YOLO App

// This class is designed to track and identify the same person across frames using the inference results of the YOLOv8-Human model in the Ultralytics YOLO app.
// The tack function is a simple tracking algorithm that tracks boxes of the same person based on box overlap across frames.
//  Access the source code: https://github.com/ultralytics/yolo-ios-app

import Accelerate
import Foundation
import Vision

class TrackingModel {
  var persons = [Person]()
  var personIndex: Int = 0
  var recent: [(CGRect, Float, [Float])] = []

  func track(boxesAndScoresAndFeatures: [(CGRect, Float, [Float])]) -> [Person] {

    if persons.isEmpty {
      for detectedHuman in boxesAndScoresAndFeatures {
        var person = Person(index: personIndex)
        person.update(box: detectedHuman.0, score: detectedHuman.1, features: detectedHuman.2)
        personIndex += 1
        persons.append(person)

      }
      return persons
    }

    var unDetectedPersonIndexes: [Int] = []
    var usedDetectedIndex: Set<Int> = Set()

    for (pi, person) in persons.enumerated() {
      var bestIOU: CGFloat = 0
      var bestIndex = 0

      for (i, detected) in boxesAndScoresAndFeatures.enumerated() {
        let IoU = overlapPercentage(rect1: person.box, rect2: detected.0)
        if IoU > bestIOU {
          bestIOU = IoU
          bestIndex = i
        }
      }
      if bestIOU >= 50 {
        let detectedPerson = boxesAndScoresAndFeatures[bestIndex]
        persons[pi].update(
          box: detectedPerson.0, score: detectedPerson.1, features: detectedPerson.2)
        usedDetectedIndex.insert(bestIndex)
      } else {
        unDetectedPersonIndexes.append(pi)
      }
    }

    let sortedIndices = unDetectedPersonIndexes.sorted(by: >)
    for index in sortedIndices {
      persons[index].unDetectedCounter += 1
    }

    for (index, det) in boxesAndScoresAndFeatures.enumerated() {
      if !usedDetectedIndex.contains(index) {
        var person = Person(index: personIndex)
        person.update(box: det.0, score: det.1, features: det.2)
        personIndex += 1
        persons.append(person)
      }
    }

    persons = removeOverlappingRects(persons: persons)

    var personsToShow: [Person] = []
    var removePersonIndexes: [Int] = []
    for (pindex, person) in persons.enumerated() {
      if person.unDetectedCounter == 0 {
        personsToShow.append(person)
      } else if person.unDetectedCounter >= 15 {
        removePersonIndexes.append(pindex)
      }
    }
    let sortedRemoveIndices = removePersonIndexes.sorted(by: >)
    for index in sortedRemoveIndices {
      persons.remove(at: index)
    }

    return personsToShow

  }
}

func overlapPercentage(rect1: CGRect, rect2: CGRect) -> CGFloat {
  let intersection = rect1.intersection(rect2)

  if intersection.isNull {
    return 0.0
  }

  let intersectionArea = intersection.width * intersection.height

  let rect1Area = rect1.width * rect1.height

  let overlapPercentage = (intersectionArea / rect1Area) * 100

  return overlapPercentage
}

func removeOverlappingRects(persons: [Person], threshold: CGFloat = 90.0) -> [Person] {
  var filteredPersons = persons
  var index = 0

  while index < filteredPersons.count {
    var shouldRemove = false
    for j in (index + 1)..<filteredPersons.count {
      let percentage = overlapPercentage(
        rect1: filteredPersons[index].box, rect2: filteredPersons[j].box)
      if percentage >= threshold {
        shouldRemove = true
        break
      }
    }
    if shouldRemove {
      filteredPersons.remove(at: index)
    } else {
      index += 1
    }
  }

  return filteredPersons
}
