<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Intelligent Interactive System Laboratory - Research</title>

  <!-- 共有ファイル -->
  <?php include($_SERVER['DOCUMENT_ROOT'].'/iis-lab/include.php'); ?>
  <!-- 個別CSS -->
  <link rel="stylesheet" type="text/css" href="/iis-lab/css/research.css">
</head>

<!-- ヘッダー埋め込み -->
<?php include($_SERVER['DOCUMENT_ROOT'].'/iis-lab/parts/en-header.php'); ?>

<body>
  <div class="main">
    <!-- articleごとに研究をまとめる -->
    <!-- 新しいのを上に -->

    <article class="research">
      <h1>Estimating Load Positions ofWearable Devices based on Difference in PulseWave Arrival Time</h1>
      <figure>
        <a href="/iis-lab/figures/research/english/ubicomp2019_yoshida.jpg" data-lightbox="group"><img src="/iis-lab/figures/research/english/ubicomp2019_yoshida.jpg"></a>
      </figure>
      <p>With the increasing use of wearable devices equipped with various sensors, human activities, biometric information, and surrounding situations can be obtained via sensor data regardless of time and place. When position-free wearable devices are attached to an arbitrary part of the body, the attached position should be identified because the application process changes relative to the position. For systems that use multiple wearable devices to capture body-wide movement, estimating the attached position of the devices is meaningful. Most conventional studies estimate the loading position of the sensor using accelerometer and gyroscope data; therefore, users must perform specific motions so that each sensor produces values unique to the given position. We propose a method that estimates the load position of wearable devices without forcing the wearer to perform specific actions. The proposed method estimates the time difference between a heartbeat obtained by an electrocardiogram and a pulse wave obtained using a pulse sensor and classifies the sensor position from the estimated time difference. We assume that pulse sensor is embedded in the wearable devices to be attached to the user. From the results of an evaluation experiment with five subjects, an average F-measure of 0.805 was achieved over 15 body parts. The left ear and the right finger achieved an F-measure of 0.9+ when the proposed system uses data of approximately 20 seconds as an input.</p><br>
      <p class="author">[UbiComp2019 / 2020 M2：Kazuki Yoshida]</p>
    </article>

    <article class="research">
      <h1>Personal identification system based on rotation of toilet paper rolls</h1>
      <figure>
        <a href="/iis-lab/figures/research/english/paper_rolls.jpg" data-lightbox="group"><img src="/iis-lab/figures/research/english/paper_rolls.jpg"></a>
      </figure>
      <p>Biological information can easily be monitored by installing sensors in a lavatory bowl. Lavatories are usually shared by several people, so users need to be identified. Because of the need for privacy, using cameras, microphones, or scales is not appropriate. Though personal identification can be done using a touch panel, the user may forget to use it because the action is not necessary. In this paper, we focus on the differences in the way of pulling a toilet paper roll and propose a system that identifies individuals based on features of rotating of toilet paper rolls with a gyroscope. The evaluation results revealed that 83.9% accuracy was achieved for a five-person group in a laboratory environment, and 69.2% accuracy was achieved for a five-person group in a practical environment.</p>
    </article>

    <article class="research">
      <h1>Activity Recognition and User Identification based on Tabletop Activities with Load Cells</h1>
      <figure>
        <a href="/iis-lab/figures/research/english/tabletop.jpg" data-lightbox="group"><img src="/iis-lab/figures/research/english/tabletop.jpg"></a>
      </figure>
      <p>There have been several studies on object detection and activity recognition on a table conducted thus far. Most of these studies use image processing with cameras or a specially configured table with electrodes and an RFID reader. In private homes, methods using cameras are not preferable since cameras might invade the privacy of inhabitants and give them the impression of being monitored. In addition, it is difficult to apply the specially configured system to off-the-shelf tables. In this work, we propose a system that recognizes activities conducted on a table and identifies which user conducted the activities with load cells only. The proposed system uses four load cells installed on the four corners of the table or under the four legs of the table. User privacy is protected because only the data on actions through the load cells is obtained. Load cells are easily installed on off-the-shelf tables with four legs and installing our system does not change the appearance of the table. The results of experiments using a table we manufactured revealed that the weight error was 38 g, the position error was 6.8 cm, the average recall of recognition for four activities was 0.96, and the average recalls of user identification were 0.65 for ten users and 0.89 for four users.</p>
    </article>

    <article class="research">
      <h1>Estimating Moving Trajectory with Sparsely Aligned Infrared Sensors in Home Environment</h1>
      <figure>
        <a href="/iis-lab/figures/research/english/home_environment.jpg" data-lightbox="group"><img src="/iis-lab/figures/research/english/home_environment.jpg"></a>
      </figure>
      <p>We propose in this paper a method for estimating trajectories of the inhabitants in a home environment, which exploits the synergy between location and movement to provide the information necessary for intelligent home appliance control. Our goal is to carry out accurate movement estimation for multiple people in a home environment. We propose an approach that uses information gathered using only passive infrared sensors commonly found in lighting control systems. No special devices or video cameras are needed. Moreover, it is not necessary to carry out data collection for training. We evaluated our approach by conducting experiments in a real home fitted with sensors and we confirmed that trajectories were detected with 0.93 recall for four inhabitants who moved upon scenarios.</p>
    </article>

    <article class="research">
      <h1>A Combined-activity Recognition Method with Accelerometers</h1>
      <figure>
        <a href="/iis-lab/figures/research/english/combined_activity.jpg" data-lightbox="group"><img src="/iis-lab/figures/research/english/combined_activity.jpg"></a>
      </figure>
      <p>Many activity recognition systems using accelerometers have been proposed. Activities that have been recognized are single activities which can be expressed with one verb, such as sitting, walking, holding a mobile phone, and throwing a ball. In fact, combined activities that include more than two kinds of state and movement are often taking place. Focusing on hand gestures, they are performed not only while standing, but also while walking and sitting. Though the simplest way to recognize such combined activities is to construct the recognition models for all the possible combinations of the activities, the number of combinations becomes immense. In this paper, firstly we propose a method that classifies activities into postures (e.g., sitting), behaviors (e.g., walking), and gestures (e.g., a punch) by using the autocorrelation of the acceleration values. Postures and behaviors are states lasting for a certain length of time. Gestures, however, are sporadic or once-off actions. It has been a challenging task to find gestures buried in other activities. Then, by utilizing the technique, we propose a recognition method for combined activities by learning single activities only. Evaluation results confirmed that our proposed method achieved 0.84 recall and 0.86 precision, which is comparable to the method that had learned all the combined activities.</p>
    </article>

    <article class="research">
      <h1>Training System of Bicycle Pedaling using Auditory Feedback</h1>
      <figure>
        <a href="/iis-lab/figures/research/english/pedaling.jpg" data-lightbox="group"><img src="/iis-lab/figures/research/english/pedaling.jpg"></a>
      </figure>
      <p>Recently, bicycling as a sport has attracted a great deal of attention. Previous research on bicycles suggests that pedaling at high frequency at a constant speed is most effective. However, it is hard for beginners to acquire such pedaling skills since expert cyclists develop the skills through long-term training. We propose a bicycle pedaling training system using auditory feedback. The system generates feedback sound every time a pedal crank turns a quarter rotation. Users can keep the pedaling speed constant by synchronizing pedaling with the feedback sound with background music whose tempo is constant. We conducted an experiment with eleven subjects for four weeks and confirmed that the variances of pedaling speed for the subjects trained with the proposed system decreased significantly compared with those of the conventional method.</p>
    </article>

    <article class="research">
      <h1>Mobile Phone User Authentication with Grip Gestures using Pressure Sensors</h1>
      <figure>
        <a href="/iis-lab/figures/research/english/grip_gestures.jpg" data-lightbox="group"><img src="/iis-lab/figures/research/english/grip_gestures.jpg"></a>
      </figure>
      <p>User authentication is generally used to protect personal information such as phone numbers, photos and account information stored in a mobile device by limiting the user to a specific person, e.g. the owner of the device. Authentication methods with password, PIN, face recognition and fingerprint identification have been widely used; however, these methods have problems of difficulty in one-handed operation, vulnerability to shoulder hacking and illegal access using fingerprint with either super glue or facial portrait. From viewpoints of usability and safety, strong and uncomplicated method is required. In this paper, a user authentication method is proposed based on grip gestures using pressure sensors mounted on the lateral and back sides of a mobile phone. Grip gesture is an operation of grasping a mobile phone, which is assumed to be done instead of conventional unlock procedure. Grip gesture can be performed with one hand. Moreover, it is hard to imitate grip gestures, as finger movements and grip force during a grip gesture are hardly seen by the others. The feature values of grip force are experimentally investigated and the proposed method from viewpoint of error rate is evaluated. From the result, this method achieved 0.02 of equal error rate, which is equivalent to face recognition. Many researches using pressure sensors to recognize grip pattern have been proposed thus far; however, the conventional works just recognize grip patterns and do not identify users, or need long pressure data to finish confident authentication. This proposed method authenticates users with a short grip gesture.</p>
    </article>

    <article class="research">
      <h1>Early Gesture Recognition Method with an Accelerometer</h1>
      <figure>
        <a href="/iis-lab/figures/research/english/accelerometer.jpg" data-lightbox="group"><img src="/iis-lab/figures/research/english/accelerometer.jpg"></a>
      </figure>
      <p>An accelerometer is installed in most current mobile phones, such as iPhones, Android-powered devices, and video game controllers for the Wii or PS3, which enables easy and intuitive operations. Therefore, many gesture-based user interfaces that use accelerometers are expected to appear in the future. Gesture recognition systems with an accelerometer generally have to be models constructed with a user's gesture data before use, and they need to recognize any unknown gestures by comparing them with an output of the recognition result and feedback delays since the recognition process generally starts after the gesture has finished, which may cause users to retry gestures and thus degrade the interface usability. We propose an early stages gesture recognition method that sequentially calculates the distance between the input and training data, and outputs recognition results only when one output candidate has a stronger likelihood than the others. Gestures are recognized in the early stages of a given motion without deteriorating the level of accuracy, which improves the interface usability. Our evaluation results indicated that the recognition accuracy approached 1.00 and the recognition results were output 1,000 msec on average before a gesture had finished.</p>
    </article>

    <article class="research">
      <h1>Determining a Number of Training Data for Gesture Recognition Considering Decay in Gesture Movements</h1>
      <figure>
        <a href="/iis-lab/figures/research/english/decay.jpg" data-lightbox="group"><img src="/iis-lab/figures/research/english/decay.jpg"></a>
      </figure>
      <p>Mobile phones and portable video games using gesture recognition technologies with an accelerometer enable drawing objects, which is difficult for conventional interfaces, and recording detailed activities in daily life. Generally, though several samples of gesture are used as training data, which may lead to misrecognition because the trajectory of gestures changes due to fatigue or forgetting gestures. However, researches considering changes of gestures have not been reported so far. We evaluate the effect of users' fatigue and forgetfulness for gesture recognition and propose a method finding appropriate position for training data in real time. We have confirmed that the proposed method finds more stable training data than that from conventional one.</p>
    </article>

    <article class="research">
      <h1>Evaluating Gesture Recognition by Multiple-Sensor-Containing Mobile Devices</h1>
      <figure>
        <a href="/iis-lab/figures/research/english/multiple_sensor_containing.jpg" data-lightbox="group"><img src="/iis-lab/figures/research/english/multiple_sensor_containing.jpg"></a>
      </figure>
      <p>In the area of activity recognition with mobile sensors, a lot of works on context-aware systems using accelerometers have been proposed. Especially, mobile phones or remotes for video games using gesture recognition technologies enable easy and intuitive operations such as scrolling browser and drawing objects. Gesture input has an advantage of rich expressive power over the conventional interfaces, but it is difficult to share the gesture motion with other people through writing or verbally. Assuming that a commercial product using gestures is released, the developers make an instruction manual and tutorial expressing the gestures in text, figures, or videos. Then an end-user reads the instructions, imagines the gesture, then perform it. In this paper, we evaluate how user gestures change according to the types of the instruction. We obtained acceleration data for 10 kinds of gestures instructed through three types of texts, figures, and videos, totalling 44 patterns from 13 test subjects, for a total of 2,630 data samples. From the evaluation, gestures are correctly performed in the order of text→figure→video. Detailed instruction in texts is equivalent to that in figures. However, some words reflecting gestures disordered the users' gestures since they could call multiple images to user's mind.</p>
    </article>

    <article class="research">
      <h1>Labeling Method for Acceleration Data using an Execution Sequence of Activities</h1>
      <figure>
        <a href="/iis-lab/figures/research/english/labeling_method.jpg" data-lightbox="group"><img src="/iis-lab/figures/research/english/labeling_method.jpg"></a>
      </figure>
      <p>In the area of activity recognition, many systems using accelerometers have been proposed. Common method for activity recognition requires raw data labeled with ground truth to learn the model. To obtain ground truth, a wearer records his/her activities during data logging through video camera or handwritten memo. However, referring a video takes long time and taking a memo interrupts natural activity. We propose a labeling method for activity recognition using an execution sequence of activities. The execution sequence includes activities in performed order, does not include time stamps, and is made based on his/her memory. Our proposed method partitions and classifies unlabeled data into segments and clusters, and assigns a cluster to each segment, then assign labels according to the best-matching assignment of clusters with the user-recorded activities. The proposed method gave a precision of 0.812 for data including seven kinds of activities. We also confirmed that recognition accuracy with training data labeled with our proposal gave a recall of 0.871, which is equivalent to that with ground truth.</p>
    </article>

    <article class="research">
      <h1>A Text Input Method for Half-Sized Keyboard using Keying Interval</h1>
      <figure>
        <img src="/iis-lab/figures/research/english/half_sized.jpg">
      </figure>
      <p>In wearable computing, compact I/O devices are desirable from the viewpoint of portability. Now, many users are accustomed to input with a keyboard, however, there is a limitation of miniaturization because it degrades the performance of key touch. Therefore, in this paper, we propose a method to miniaturize a keyboard by excluding the half of it. The user can input words with one hand because the proposed system estimates the input word using keying interval, which appears also when the user inputs with both hands. From the result of user study, we confirmed that the user can input with only one hand and that it does not decrease input speed drastically.</p>
    </article>

    <article class="research">
      <h1>A Shape Command Input Method using Key Entry Information of Physical Keyboard</h1>
      <figure>
        <a href="/iis-lab/figures/research/english/shape_command.jpg" data-lightbox="group"><img src="/iis-lab/figures/research/english/shape_command.jpg"></a>
      </figure>
      <p>Keyboard is mainly used for text input, and has function keys and shortcut keys to reduce the number of manual operation. However, it is difficult and troublesome to memorize relations between keys and functions for beginners. In this paper, we propose two intuitional input methods on the physical keyboard in addition to usual text-typing: stroke that traces a shape on the keyboard, and stamp that presses a shape on the keyboard. Our system automatically classifies user inputs into text-typing, stroke, or stamp from key entry information, enabling us to seamlessly input those commands without additional devices.</p>
    </article>

    <article class="research">
      <h1>A Method for Energy Saving on Context-aware System by Sampling Control and Data Complement</h1>
      <figure>
        <a href="/iis-lab/figures/research/english/energy_saving.jpg" data-lightbox="group"><img src="/iis-lab/figures/research/english/energy_saving.jpg"></a>
      </figure>
      <p>The downsizing of computers has led to wearable computing that has attracted a great deal of attention. In wearable computing environments, a wearable computer runs various applications using various wearable sensors. In the research area of context-awareness, though various systems use multiple accelerometers to recognize minute motions and states, conventional architecture has a room to be optimized from the viewpoint of energy consumption and accuracy. In this paper, we propose a context-aware system that reduces energy consumption by controlling the sampling frequency of wearable sensors. Even if the sampling frequency changes, no extra configurations on recognition and learning algorithm are required because the missing data for controlled sensors are complemented by our proposed algorithm. By using our system, energy consumption can be reduced without large loss in accuracy.</p>
    </article>

    <article class="research">
      <h1>A Motion Recognition Method by Constancy Decision</h1>
      <figure>
        <a href="/iis-lab/figures/research/english/constancy.jpg" data-lightbox="group"><img src="/iis-lab/figures/research/english/constancy.jpg"></a>
      </figure>
      <p>The downsizing of computers has led to wearable computing that has attracted a great deal of attention.In the area of context awareness, many context-aware systems using accelerometers have been proposed. Contexts that have been recognized are categorized into postures (e.g., sitting), behaviors (e.g., walking), and gestures (e.g., draw a circle). Postures and behaviors are states lasting for a certain length of time, which are recognized with several feature values over a window. Gestures, however, are once-off actions. It has been a challenging task to find gestures on real environments where gestures are buried in other contexts. In this paper, we propose a method that classifies contexts into postures, behaviors, and gestures by using the autocorrelation of the acceleration values and recognizes contexts with an appropriate method. We evaluated the performance of recognition for seven kinds of gestures while five kinds of behaviors; The conventional method gave recall and precision of 0.75 and 0.59 whereas our method gave 0.93 and 0.92, respectively. Our system enables a user to input by gesturing even while he or she is performing a behavior.</p>
    </article>

    <article class="research">
      <h1>A Method for Context Recognition Using Peak Values of Sensors</h1>
      <figure>
        <a href="/iis-lab/figures/research/english/peak.jpg" data-lightbox="group"><img src="/iis-lab/figures/research/english/peak.jpg"></a>
      </figure>
      <p>In wearable computing environments, various applications are assumed to get a richer sense of context via a set of wearable sensors. When obtaining the wearer's context, raw sensor values typically have to be pre-processed before recognition can take place. This process of feature-extraction in wearable sensing has thus far favored combinations of mean, variance, and Fourier coefficients over a sliding window as highly-discriminative features and have been used extensively so far in the literature. Since the size of the features tends to become larger than that of the raw data itself, sensors send raw data to a main computer, then feature-extraction and recognition take place. However, raw data consume large power for wireless communication and writing to their memories, conflicting with the often low-power hardware in wearable computing. In this research, we suggest width and height of peaks as features that perform in the range of conventional features but that have smaller data size. By using our proposal, sensors shrink data and send these to the main computer after feature-extraction, which would conserve power.</p>
    </article>

    <article class="research">
      <h1>Navigation System with a Route Planning Algorithm Using Body-worn Sensors</h1>
      <figure>
        <a href="/iis-lab/figures/research/english/route_planning.jpg" data-lightbox="group"><img src="/iis-lab/figures/research/english/route_planning.jpg"></a>
      </figure>
      <p>There are many kinds of event spaces such as stamp rally and amusement spot. In these events, the participants behave as they want and it causes problems for the event managers, such as too long necessary time or the congestion of specific attractions. In this paper, we propose a navigation system that has a route planning algorithm to satisfy the purposes for the event manager. The result of actual use in our system bore out that the participants behaved according to the purposes of the event manager. Moreover, by using a wearable computer and wearable sensors, participants' real-time information are acquired and our system performs better.</p>
    </article>

    <article class="research">
      <h1>Context-aware System Considering Energy Consumption for Wearable Computing</h1>
      <figure>
        <img src="/iis-lab/figures/research/english/context_aware.jpg">
      </figure>
      <p>In wearable computing environments, a wearable computer runs various applications using various wearable sensors. In the area of context awareness, though various systems use multiple accelerometers to recognize very minute motions and states, energy consumption was not taken into consideration. We propose a context-aware system that reduces energy consumption. The proposed system changes sensor combination in terms of energy consumption and accuracy, and turns unused sensors off. Even if the number of sensors changes, no extra classifiers or training data are required because the data for shutting off sensors is complemented by our proposed algorithm. By using our system, power consumption can be reduced without large losses in accuracy.</p>
    </article>

    <article class="research">
      <h1>CLAD: Cross-linkage for Assembled Devices</h1>
      <figure>
        <img src="/iis-lab/figures/research/english/clad.gif">
      </figure>
      <p>In wearable computing environments, a wearable computer runs various applications with various sensors (wearable sensors). Since conventional wearable systems do not manage the power supply flexibly, they consume excess power resource for unused sensors. Additionally, sensors frequently become unstable by several reasons such as breakdown of sensors. It is inadequate for application engineers to detect them only by sensing data. To solve these problems, we propose a new sensor management device CLAD (Cross-Linkage for Assembled Devices) that has various functions for power management and sensed data management. CLAD improves power saving, data accuracy, and operational reliability.</p>
    </article>

    <article class="research">
      <h1>VegeTongs: Vegetable Recognition Tongs using Active Acoustic Sensing</h1>
      <figure>
        <img src="/iis-lab/figures/research/english/vegetongs.jpg">
      </figure>
      <p>People have been engaged in production activities such as food, clothing and housing using tools in many situations. In such a situation, it is useful to be able to recognize an object with which a person interacts Therefore, we propose a method to perform object recognition effectively by applying active acoustic sensing technology to the user's tool when the user interacts with the object through the tool Using this method, we implemented the tongs-type device that sandwiches the ingredients, targeting the situatuion where the ingredients are interacted with using tools such as cooking and eating. As a result of evaluation experiment of the proposed system, it was possible to recognize at 88%, and it was confirmed that the proposed system was effective.</p>
    </article>
  </div>
</body>

<!-- フッター埋め込み -->
<?php include($_SERVER['DOCUMENT_ROOT'].'/iis-lab/parts/en-footer.php'); ?>
</html>
