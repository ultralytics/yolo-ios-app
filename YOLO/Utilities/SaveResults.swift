//
//  SaveResults.swift
//  YOLO
//
//  Created by 間嶋大輔 on 2024/06/26.
//  Copyright © 2024 Ultralytics. All rights reserved.
//

import Foundation

let detectionHeader = "sec_day, free_space, batteryLevel ,class,confidence,box\n"
let humanHeader = "sec_day, free_space, battery_level ,id, confidence, box_x, box_y, box_w, box_h, weight, height, age, gender, gender_confidence, race, race_confidence \n"

func saveDetectionResultsToCSV(detectionResults:[String], task: Task) -> URL? {
    var header = ""
    var taskName = ""
    switch task {
    case .detect:
        header = detectionHeader
        taskName = "detection"

    case .human:
        header = humanHeader
        taskName = "human"
    }
    let formatter = DateFormatter()
    formatter.dateFormat = "yyyyMMdd_HH:mm:ss"
    let dateString = formatter.string(from: Date())
    let fileName =  taskName + "_results_\(dateString).csv"

    let path = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent(fileName)
    
    var csvText = header
    
    for result in detectionResults {
        csvText.append(contentsOf: result)
    }
    
    do {
        try csvText.write(to: path, atomically: true, encoding: .utf8)
        print("CSV file saved at: \(path)")
        return path
    } catch {
        print("Failed to save CSV file: \(error)")
        return nil
    }
}
