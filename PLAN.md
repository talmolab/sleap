# Video Player Threading Architecture Fix Plan

## Context and Problem Analysis

### The Issue
When scrubbing through video frames in the SLEAP GUI, we encounter the error:
```
QBasicTimer::start: Timers cannot be started from another thread
```

This error originates from Qt's internal timer management and indicates a fundamental threading architecture problem in our video player implementation.

### Current Architecture Problems

#### 1. Multiple Concurrent Timers
- **MainWindow** has `update_gui_timer` running `_update_gui_state()` 
- **LoadImageWorker** has a QTimer triggering `doProcessing()` every 20ms
- **QApplication** main event loop running `exec_()`
- These timers can conflict when signals cross thread boundaries

#### 2. Complex Signal/Slot Chain for Frame Rendering

The current call chain when user changes frames:

```
User scrubs seekbar
    ↓
seekbar.valueChanged signal
    ↓
state.set("frame_idx", value)  [Main Thread]
    ↓
GuiState.emit("frame_idx")  [Main Thread]
    ↓
Multiple callbacks triggered:
    - QtVideoPlayer.plot() via lambda
    - seekbar.setValue() via lambda
    ↓
QtVideoPlayer.plot()  [Main Thread]
    ↓
1. view.clear()
2. changedPlot.emit(self, idx, instance)
3. load_image_worker.request(video, idx)
    ↓
frame_requested.emit(video, frame_idx)  [Main → Worker Thread]
    ↓
LoadImageWorker._handle_request()  [Worker Thread]
    ↓
If FORCE_REQUESTS and timeout:
    force_process.emit()  [Worker Thread]
    ↓
LoadImageWorker.doProcessing()  [Worker Thread - CONFLICT!]
```

#### 3. The Core Problem

The `QBasicTimer` error occurs because:
- `LoadImageWorker` has a QTimer that calls `doProcessing()` every 20ms in the worker thread
- When `force_process.emit()` also triggers `doProcessing()`, it can cause the timer's internal state to be manipulated incorrectly
- Qt's QTimer uses QBasicTimer internally, and having multiple paths to the same slot from different thread contexts causes thread affinity issues
- The timer might be trying to restart itself while already running, or being accessed from the wrong thread

#### 4. Additional Complexities

1. **Circular Dependencies**: 
   - `MainWindow._after_plot_change()` can set `state["frame_idx"]` again, potentially causing infinite loops
   - Multiple state changes can trigger cascading plot() calls

2. **Thread Safety Issues**:
   - `load_queue` is a regular Python list, not thread-safe
   - Video object is passed between threads
   - QImage/QPixmap memory management across threads

3. **Race Conditions**:
   - Rapid scrubbing creates many pending requests
   - Timer and signal-based processing can collide
   - Mutex locking doesn't prevent timer conflicts

## Detailed Implementation Plan

### Phase 1: Comprehensive Debugging and Analysis
**Goal**: Understand the exact call chain and thread contexts where the error occurs.

#### Step 1.1: Add Thread Tracking
Add debug logging to track thread IDs at critical points:

```python
# In LoadImageWorker methods
def start_work(self):
    print(f"[WORKER] start_work thread: {QtCore.QThread.currentThread()} id:{id(QtCore.QThread.currentThread())}")
    
def doProcessing(self):
    print(f"[WORKER] doProcessing thread: {QtCore.QThread.currentThread()} id:{id(QtCore.QThread.currentThread())}")
    print(f"[WORKER] doProcessing called, queue size: {len(self.load_queue)}")
    
def _handle_request(self, video, frame_idx):
    print(f"[WORKER] _handle_request thread: {QtCore.QThread.currentThread()} id:{id(QtCore.QThread.currentThread())}")
```

#### Step 1.2: Track Signal Emissions
Log all signal emissions and their sources:

```python
# In QtVideoPlayer
def plot(self):
    print(f"[MAIN] plot() called, thread: {QtCore.QThread.currentThread()}")
    print(f"[MAIN] plot() frame_idx: {self.state['frame_idx']}")
    
# In GuiState
def emit(self, key):
    if key == "frame_idx":
        print(f"[STATE] emit('frame_idx') with value: {self.get(key)}")
        print(f"[STATE] callbacks for frame_idx: {len(self._callbacks.get(key, []))}")
```

#### Step 1.3: Monitor Timer State
Track timer operations:

```python
# In LoadImageWorker
def start_work(self):
    if self.timer is None:
        print("[TIMER] Creating new QTimer")
        self.timer = QtCore.QTimer()
        print(f"[TIMER] Timer thread affinity: {self.timer.thread()}")
    else:
        print("[TIMER] Timer already exists!")
        
# Before timer.start()
print(f"[TIMER] Starting timer, isActive: {self.timer.isActive()}")

# In stop_work()
print(f"[TIMER] Stopping timer, isActive: {self.timer.isActive() if self.timer else 'None'}")
```

### Phase 2: Remove QTimer and Implement Signal-Based Processing

#### Step 2.1: Remove QTimer from LoadImageWorker
Replace the timer-based approach with pure signal-driven processing:

```python
class LoadImageWorker(QtCore.QObject):
    result = QtCore.Signal(QImage)
    frame_requested = QtCore.Signal(object, int)
    
    def __init__(self):
        super().__init__()
        self.load_queue = []
        self.current_video = None
        self._processing_mutex = QtCore.QMutex()
        self.is_running = False
        self._is_processing = False
        
    @QtCore.Slot()
    def start_work(self):
        self.is_running = True
        self._processing_mutex = QtCore.QMutex()
        
        # Connect signal to slot with QueuedConnection
        self.frame_requested.connect(
            self._handle_request,
            QtCore.Qt.QueuedConnection
        )
        
    @QtCore.Slot(object, int)
    def _handle_request(self, video, frame_idx):
        if not self.is_running:
            return
            
        # Update video reference
        self.current_video = video
        
        # Clear old requests and add new one (FILO)
        self.load_queue = [frame_idx]
        
        # Process immediately
        self._process_frame()
        
    def _process_frame(self):
        if self._is_processing:
            return  # Already processing
            
        if not self._processing_mutex.tryLock(100):
            return  # Couldn't get lock
            
        try:
            self._is_processing = True
            
            if not self.load_queue or not self.current_video:
                return
                
            frame_idx = self.load_queue[-1]
            self.load_queue = []
            
            # Load frame
            frame = self.current_video.get_frame(frame_idx)
            
            if frame is not None:
                qimage = ndarray_to_qimage(frame, copy=True)
                self.result.emit(qimage)
                
        except Exception as e:
            print(f"[ERROR] Frame loading failed: {e}")
        finally:
            self._is_processing = False
            self._processing_mutex.unlock()
```

#### Step 2.2: Implement Request Throttling
Add a mechanism to throttle rapid requests without using timers:

```python
class LoadImageWorker(QtCore.QObject):
    def __init__(self):
        # ... existing init ...
        self._last_request_time = 0
        self._min_request_interval = 0.016  # ~60 FPS max
        
    @QtCore.Slot(object, int)
    def _handle_request(self, video, frame_idx):
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self._min_request_interval:
            # Defer processing
            QtCore.QTimer.singleShot(
                int((self._min_request_interval - time_since_last) * 1000),
                lambda: self._process_frame()
            )
        else:
            self._process_frame()
            
        self._last_request_time = current_time
```

### Phase 3: Fix Thread Architecture

#### Step 3.1: Ensure Proper Thread Affinity
Make sure all Qt objects are created in the correct threads:

```python
class QtVideoPlayer(QWidget):
    def _setup_worker_thread(self):
        # Create thread in main thread
        self.load_image_worker_thread = QtCore.QThread()
        
        # Create worker in main thread
        self.load_image_worker = LoadImageWorker()
        
        # Move worker to thread BEFORE connections
        self.load_image_worker.moveToThread(self.load_image_worker_thread)
        
        # Use QueuedConnection for all cross-thread signals
        self.load_image_worker.result.connect(
            self._on_frame_loaded,
            QtCore.Qt.QueuedConnection
        )
        
        # Start work when thread starts
        self.load_image_worker_thread.started.connect(
            self.load_image_worker.start_work,
            QtCore.Qt.DirectConnection  # Same thread
        )
        
        # Start thread
        self.load_image_worker_thread.start()
        
    @QtCore.Slot(QImage)
    def _on_frame_loaded(self, qimage):
        """Handle loaded frame in main thread"""
        if self.view:
            self.view.setImage(qimage)
```

#### Step 3.2: Prevent Circular Signal Loops
Add guards to prevent infinite loops:

```python
class QtVideoPlayer(QWidget):
    def __init__(self):
        # ... existing init ...
        self._is_plotting = False
        
    def plot(self):
        if self._is_plotting:
            return  # Prevent re-entry
            
        self._is_plotting = True
        try:
            # ... existing plot logic ...
        finally:
            self._is_plotting = False
```

### Phase 4: Testing Strategy

Since GUI testing requires user interaction, here's the testing approach:

#### Test 4.1: Basic Frame Loading
**User Action Required**: 
1. Open a video file
2. Click on different frames slowly (1 frame per second)
3. Verify no errors in console
4. Verify frames display correctly

#### Test 4.2: Rapid Scrubbing
**User Action Required**:
1. Open a video file
2. Rapidly drag the seekbar back and forth
3. Monitor console for QBasicTimer errors
4. Verify no crashes or freezes
5. Verify final frame displays correctly

#### Test 4.3: Stress Testing
**User Action Required**:
1. Open a large video file (>1000 frames)
2. Use keyboard shortcuts to rapidly advance frames
3. Test jump to beginning/end buttons
4. Test frame range selection and scrubbing within range

#### Test 4.4: Thread Safety
**User Action Required**:
1. Open multiple videos in sequence
2. Switch between videos while scrubbing
3. Close and reopen GUI multiple times
4. Monitor for segfaults or exit code 139

#### Test 4.5: Memory Leaks
**User Action Required**:
1. Monitor memory usage before opening video
2. Scrub through entire video multiple times
3. Close video and check memory is released
4. Repeat with different video formats

### Phase 5: Validation and Monitoring

#### Step 5.1: Add Performance Metrics
```python
class LoadImageWorker(QtCore.QObject):
    def __init__(self):
        # ... existing init ...
        self._frame_load_times = deque(maxlen=100)
        self._dropped_frames = 0
        
    def _process_frame(self):
        start_time = time.time()
        # ... process frame ...
        load_time = time.time() - start_time
        self._frame_load_times.append(load_time)
        
        if len(self.load_queue) > 0:
            self._dropped_frames += len(self.load_queue)
            print(f"[PERF] Dropped {len(self.load_queue)} frames")
            
        if len(self._frame_load_times) == 100:
            avg_time = sum(self._frame_load_times) / 100
            print(f"[PERF] Avg frame load time: {avg_time:.3f}s")
```

#### Step 5.2: Add Error Recovery
```python
class LoadImageWorker(QtCore.QObject):
    error_occurred = QtCore.Signal(str)
    
    def _process_frame(self):
        try:
            # ... existing processing ...
        except Exception as e:
            error_msg = f"Frame loading error: {str(e)}"
            print(f"[ERROR] {error_msg}")
            self.error_occurred.emit(error_msg)
            # Don't crash, just skip this frame
```

### Implementation Order

1. **First Session**: 
   - Implement Phase 1 debugging
   - Ask user to test and collect logs
   
2. **Second Session**:
   - Based on logs, implement Phase 2 (remove QTimer)
   - Ask user to test basic functionality
   
3. **Third Session**:
   - Implement Phase 3 (thread architecture fixes)
   - Ask user for comprehensive testing
   
4. **Fourth Session**:
   - Based on test results, implement optimizations
   - Add performance monitoring
   - Final validation

### Success Criteria

1. No `QBasicTimer::start` errors during rapid scrubbing
2. No segfaults or exit code 139
3. Smooth frame rendering without glitches
4. No memory leaks after extended use
5. Frame loading performance ≤ 50ms for standard videos

### Rollback Plan

If the new architecture causes issues:
1. Keep the old implementation in a separate file
2. Add a feature flag to switch between implementations
3. Gradually migrate features once stable
4. Maintain backwards compatibility with existing code

### Notes for Implementation

- Always use `QtCore.Qt.QueuedConnection` for cross-thread signals
- Never access GUI elements from worker thread
- Use `QMutexLocker` for RAII-style locking when possible
- Consider using `QThreadPool` for future scalability
- Document all thread affinities in code comments