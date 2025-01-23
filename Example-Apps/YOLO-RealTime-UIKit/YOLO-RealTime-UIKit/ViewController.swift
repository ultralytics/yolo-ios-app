
import UIKit
import YOLO
class ViewController: UIViewController {

    var yoloView:YOLOView!
    override func viewDidLoad() {
        super.viewDidLoad()
        yoloView = YOLOView(frame: view.bounds, modelPathOrName: "yolo11n-cls", task:.classify)
        view.addSubview(yoloView)
    }
}

