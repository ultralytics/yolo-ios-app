//  Ultralytics YOLO 🚀 - AGPL-3.0 License
//
//  Colors for Ultralytics YOLO App

// These colors are set to use the same Ultralytics color palette as the Python version.

import Foundation
import UIKit

let ultralyticsColorsolors: [UIColor] = [
  UIColor(red: 4 / 255, green: 42 / 255, blue: 255 / 255, alpha: 0.6),  // #042AFF
  UIColor(red: 11 / 255, green: 219 / 255, blue: 235 / 255, alpha: 0.6),  // #0BDBEB
  UIColor(red: 243 / 255, green: 243 / 255, blue: 243 / 255, alpha: 0.6),  // #F3F3F3
  UIColor(red: 0 / 255, green: 223 / 255, blue: 183 / 255, alpha: 0.6),  // #00DFB7
  UIColor(red: 17 / 255, green: 31 / 255, blue: 104 / 255, alpha: 0.6),  // #111F68
  UIColor(red: 255 / 255, green: 111 / 255, blue: 221 / 255, alpha: 0.6),  // #FF6FDD
  UIColor(red: 255 / 255, green: 68 / 255, blue: 79 / 255, alpha: 0.6),  // #FF444F
  UIColor(red: 204 / 255, green: 237 / 255, blue: 0 / 255, alpha: 0.6),  // #CCED00
  UIColor(red: 0 / 255, green: 243 / 255, blue: 68 / 255, alpha: 0.6),  // #00F344
  UIColor(red: 189 / 255, green: 0 / 255, blue: 255 / 255, alpha: 0.6),  // #BD00FF
  UIColor(red: 0 / 255, green: 180 / 255, blue: 255 / 255, alpha: 0.6),  // #00B4FF
  UIColor(red: 221 / 255, green: 0 / 255, blue: 186 / 255, alpha: 0.6),  // #DD00BA
  UIColor(red: 0 / 255, green: 255 / 255, blue: 255 / 255, alpha: 0.6),  // #00FFFF
  UIColor(red: 38 / 255, green: 192 / 255, blue: 0 / 255, alpha: 0.6),  // #26C000
  UIColor(red: 1 / 255, green: 255 / 255, blue: 179 / 255, alpha: 0.6),  // #01FFB3
  UIColor(red: 125 / 255, green: 36 / 255, blue: 255 / 255, alpha: 0.6),  // #7D24FF
  UIColor(red: 123 / 255, green: 0 / 255, blue: 104 / 255, alpha: 0.6),  // #7B0068
  UIColor(red: 255 / 255, green: 27 / 255, blue: 108 / 255, alpha: 0.6),  // #FF1B6C
  UIColor(red: 252 / 255, green: 109 / 255, blue: 47 / 255, alpha: 0.6),  // #FC6D2F
  UIColor(red: 162 / 255, green: 255 / 255, blue: 11 / 255, alpha: 0.6),  // #A2FF0B
]

let posePalette: [[CGFloat]] = [
  [255, 128, 0],
  [255, 153, 51],
  [255, 178, 102],
  [230, 230, 0],
  [255, 153, 255],
  [153, 204, 255],
  [255, 102, 255],
  [255, 51, 255],
  [102, 178, 255],
  [51, 153, 255],
  [255, 153, 153],
  [255, 102, 102],
  [255, 51, 51],
  [153, 255, 153],
  [102, 255, 102],
  [51, 255, 51],
  [0, 255, 0],
  [0, 0, 255],
  [255, 0, 0],
  [255, 255, 255],
]

let limbColorIndices = [0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16]
let kptColorIndices = [16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0]

let skeleton = [
  [16, 14],
  [14, 12],
  [17, 15],
  [15, 13],
  [12, 13],
  [6, 12],
  [7, 13],
  [6, 7],
  [6, 8],
  [7, 9],
  [8, 10],
  [9, 11],
  [2, 3],
  [1, 2],
  [1, 3],
  [2, 4],
  [3, 5],
  [4, 6],
  [5, 7],
]
