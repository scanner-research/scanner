var Immutable = require('immutable');
var React = require('react')
  var ReactDOM = require('react-dom');
var _ = require('lodash');
var $ = require('jquery');
var update = require('react-addons-update');
var Video = require('react-html5video/dist/ReactHtml5Video').default;


require("../css/index.scss");

var item_bbox = {
  frame: 100,
  time: 36,
  data: {
    bboxes: [
      {
        x: 100,
        y: 125,
        width: 50,
        height: 20,
      },
      {
        x: 200,
        y: 75,
        width: 28,
        height: 60,
      },
    ],
  },
};

var classificationGraphics = {
  setup: function(mainPanel) {
  },
  plotValues: function(sampledData) {
    var confidenceData = [];
    for (var i = 0; i < sampledData.length; ++i) {
      var item = sampledData[i];
      // var norm = 0;
      // for (var j = 0; j < item.data.length; ++j) {
      //     norm += item.data[j] * item.data[j];
      //     if (item.data[j] > max) {
      //         max = item.data[j];
      //     }
      // }
      // var max = 0.0;
      // for (var j = 0; j < item.data.length; ++j) {
      //     if (item.data[j] / norm > max) {
      //         max = item.data[j] / norm;
      //     }
      // }
      // confusionData.append({
      //     frame: i,
      //     confusion: max
      // });
      confidenceData.push({
        frame: item.frame,
        value: item.data.confidence,
      });
    }
    return confidenceData;
  },
  show: function() {
    var classIndicator = $("#class-indicator");
    classIndicator.show();
  },
  draw: function(videoMetadata, item) {
    var classIndicator = $("#class-indicator");
    classIndicator.text(item.data.confidence + "\n" +
                        labelNames[item.data["class"]]);
  },
  hide: function() {
    var classIndicator = $("#class-indicator");
    classIndicator.hide();
  },
};

var detectionGraphics = {
  setup: function(container, videoElement) {
    var container = $(container);
    var videoElement = $(videoElement);
    container.append();
    detectionGraphics.svgContainer = d3.select(container.get(0))
                                       .append("svg")
                                       .classed("bbox-container", true)
                                       .attr("width", videoElement.width())
                                       .attr("height", videoElement.height())
                                       .append("g");
  },
  teardown: function(container) {
  },
  show: function() {
  },
  draw: function(videoElement, videoMetadata, item) {
    var videoElement = $(videoElement);
    var bboxes = detectionGraphics.svgContainer.selectAll(".bbox")
                                  .data(item.data.base_bboxes, function (d, i) {
                                    var id = item.frame + ':' + i;
                                    return id;
                                  });
    var w = videoMetadata.width;
    var h = videoMetadata.height;
    var viewWidth = videoElement.width();
    var viewHeight = videoElement.height();
    bboxes.enter()
          .append("rect")
          .attr("class", "bbox")
          .attr("x", function(d) {
            return (d.x - d.width / 2) / w * viewWidth;
          })
          .attr("y", function(d) {
            return (d.y - d.height / 2) / h * viewHeight;
          })
          .attr("width", function(d) {
            return d.width / w * viewWidth;
          })
          .attr("height", function(d) {
            return d.height / h * viewHeight;
          })
          .style("stroke", function (d) {
            var color;
            if (d.category == 0) {
              color = 'red';
            } else if (d.category == 1) {
              color = 'green';
            } else {
              color = 'blue';
            }
            return color;
          });
    bboxes
          .attr("x", function(d) {
              return (d.x - d.width / 2) / w * viewWidth;
          })
          .attr("y", function(d) {
              return (d.y - d.height / 2) / h * viewHeight;
          })
          .attr("width", function(d) {
              return d.width / w * viewWidth;
          })
          .attr("height", function(d) {
              return d.height / h * viewHeight;
          })
          .style("stroke", function (d) {
              var color;
              if (d.category == 0) {
                  color = 'red';
              } else if (d.category == 1) {
                  color = 'green';
              } else {
                  color = 'blue';
              }
              return color;
          });
      bboxes.exit()
            .remove();
  },
  hide: function() {
    // detectionGraphics.svgContainer.selectAll("rect.bbox")
    //     .data([])
    //     .exit()
    //     .remove();
  },
  //
};

function drawBbox() {
}

var graphicsOptions = {
  "classification": classificationGraphics,
  "detection": detectionGraphics,
};

function draw(v,c) {
  c.drawImage(v,0,0,c.canvas.width,c.canvas.height);
}

function setupViewer(container, mainPanel, jobMetadata, videoMetadata) {
  var viewer = $('<div/>', {'class': 'video-viewer'});
  container.append(viewer)

    var video = $('<video/>', {'width': viewWidth,
                               'height': viewHeight,
                               'class': 'video-viewer',
                               'controls': 'on'})
        .hide();
  viewer.append(video);

  /*var videoSource = $('<source/>', {'src': videoMetadata.mediaPath,
     'type': 'video/mp4'});
     video.append(videoSource);*/

  var timeline = $('<div/>', {'class': 'video-timeline'});
  viewer.append(timeline);
  var timelineThumbnails = $('<div/>', {'class': 'video-thumbnails'});
  timeline.append(timelineThumbnails);

  var htmlVideo = video[0];

  var totalDuration = 0;
  $(htmlVideo).on("loadedmetadata", function() {
    totalDuration = this.duration;
    htmlVideo.currentTime = 1;
  });

  var thumbnailsContainer = timelineThumbnails;
  var thumbnailWidth = 100;
  var thumbnailHeight = 56.26;
  var totalThumbnails =
  Math.floor(thumbnailsContainer.width() / thumbnailWidth);

  //var currentThumbnail = 0;
  var currentThumbnail = totalThumbnails;
  for (var i = 0; i < totalThumbnails; ++i) {
    var thumbnailCanvas = $('<canvas/>',{'class':'thumbnail'})
            .width(thumbnailWidth)
            .height(thumbnailHeight);
    thumbnailsContainer.append(thumbnailCanvas);
  }

  $(htmlVideo).on("seeked", function() {
    if (currentThumbnail < totalThumbnails) {
      var thumbnailContext = thumbnailsContainer
                .children()[currentThumbnail]
                .getContext("2d");
      draw(this, thumbnailContext);

      currentThumbnail += 1;
      this.currentTime =
      (totalDuration / (totalThumbnails)) * currentThumbnail;
    }
  });

  setupTimeline(timeline, mainPanel, htmlVideo, jobMetadata, videoMetadata);
}

function setupTimeline(container,
                       mainPanel,
                       video,
                       jobMetadata,
                       videoMetadata)
{
  var requestRadius = 1;
  var stride = 1;

  var mainVideo = mainPanel.children("#main-viewer")[0];
  var mainVideoSource = $(mainVideo).children("source");
  var frameIndicator = mainPanel.children("#frame-indicator");
  var classIndicator = mainPanel.children("#class-indicator");

  $(mainVideo).attr('width', viewWidth);
  $(mainVideo).attr('height', viewHeight);

  var width = container.width();
  var tickWidth = 100;
  var tickHeight = 80;
  var ticks = Math.floor(width / tickWidth);

  var hoveredFrame = -1;
  var selectedFrame = -1;
}

function setupTimelinePlot(axis,
                           video,
                           jobMetadata,
                           videoMetadata,
                           plotWidth,
                           plotHeight,
                           predictionData)
{
  var plotCanvas = $('<canvas/>', {'class': 'timeline-plot',
                                   'width': plotWidth,
                                   'height': plotHeight})
        .css('width', plotWidth)
        .css('height', plotHeight);
  axis.append(plotCanvas)

    plotCanvas[0].width = plotWidth;
  plotCanvas[0].height = plotHeight;
  var context = plotCanvas[0].getContext("2d");

  var margin = {top: 0, right: 0, bottom: 0, left: 0},
      canvasWidth = plotWidth - margin.left - margin.right,
      canvasHeight = plotHeight - margin.top - margin.bottom;

  var x = d3.scaleLinear()
            .range([0, canvasWidth]);

  var y = d3.scaleLinear()
            .range([canvasHeight, 0]);

  var line = d3.line()
               .x(function(d) { return x(d.frame); })
               .y(function(d) { return y(d.value); })
               .curve(d3.curveStep)
               .context(context);

  context.translate(margin.left, margin.top);

  var lineData = jobMetadata.graphics.plotValues(predictionData);

  //x.domain(d3.extent(confusionData, function(d) { return d.frame; }));
  x.domain([0, videoMetadata.frames]);
  y.domain([0, 1]);
  //y.domain(d3.extent(lineData, function(d) { return d.value; }));
  context.beginPath();
  line(lineData);
  context.lineWidth = 1.5;
  context.strokeStyle = "steelblue";
  context.stroke();
}

var VideoTimeline = React.createClass({
  getInitialState: function() {
    return {
      width: 0,
      selectedFrame: -1,
    };
  },
  posToFrameNumber: function(pageX) {
    var axis = $(this.refs.axis);
    var offset = axis.offset();
    var xPos = pageX - offset.left;
    var percent = xPos / axis.width();
    var frame = Math.floor(this.props.video.frames * percent);
    return frame;
  },
  handleMouseMove: function(e) {
    var targetedFrame = this.posToFrameNumber(e.pageX);

    this.state.onSelectedFrameChange({
      videoId: this.props.video.id,
      frame: targetedFrame,
    });
  },
  handleClick: function(e) {
    var targetedFrame = this.posToFrameNumber(e.pageX);

    this.state.onSelectedFrameChange({
      videoId: this.props.video.id,
      frame: targetedFrame,
    });
    this.setState({selectedFrame: targetedFrame});
  },
  handleMouseLeave: function(e) {
    this.state.onSelectedFrameChange({
      videoId: this.props.video.id,
      frame: this.state.selectedFrame,
    });
  },
  componentDidMount: function() {
    var width = $(ReactDOM.findDOMNode(this)).width();
    var onSelectedFrameChange =
    _.debounce(this.props.onSelectedFrameChange, 50);
    this.setState({
      width: width,
      onSelectedFrameChange: onSelectedFrameChange
    });
  },
  componentWillReceiveProps: function(nextProps) {
    var onSelectedFrameChange =
    _.debounce(this.props.onSelectedFrameChange, 50);
    this.setState({
      onSelectedFrameChange: onSelectedFrameChange
    });
  },
  render: function() {
    var video = this.props.video;

    var tickWidth = 100;
    var tickHeight = 80;

    var labelWidth = 50;
    var labelHeight = 20;

    var numTicks = Math.floor(this.state.width / tickWidth);
    var ticks = _.times(numTicks, function(i) {
      var style = {
        left: tickWidth * i,
        width: tickWidth - 1,
        top: 0,
        height: tickHeight,
      };
      return (
        <div className="timeline-tick"
             style={style}
             key={i}>
        </div>
      );
    });
    var style = {
      left: tickWidth * numTicks - 1,
      width: 0,
      top: 0,
      height: tickHeight,
    };
    var lastTick = (
      <div className="timeline-tick"
           style={style}
           key={numTicks}>
      </div>
    );
    ticks.push(lastTick);
    var tickLabels = _.times(numTicks, function(i) {
      var style = {
        left: tickWidth * i - labelWidth / 2,
        width: labelWidth,
        top: tickHeight,
        height: labelHeight,
      };
      return (
        <div className="timeline-tick-label"
             style={style}
             key={i}>
          {Math.round(video.frames / numTicks * i)}
        </div>
      );
    });
    var style = {
      left: tickWidth * numTicks - labelWidth / 2,
      width: labelWidth,
      top: tickHeight,
      height: labelHeight,
    };
    var lastTickLabel = (
      <div className="timeline-tick-label"
           style={style}
           key={numTicks}>
        {video.frames}
      </div>
    );
    tickLabels.push(lastTickLabel);

    return (
      <div className="video-timeline"
           onClick={this.handleClick}
           onMouseMove={this.handleMouseMove}
           onMouseLeave={this.handleMouseLeave}>
        <div className="timeline-axis"
             ref="axis">
          {ticks}
          {tickLabels}
        </div>
      </div>
    )
  }
});

var VideoNavigator = React.createClass({
  render: function() {
    return (
      <div className="video-navigator">
        <VideoTimeline job={this.props.job}
                       video={this.props.video}
                       onSelectedFrameChange={
                                              this.props.onSelectedFrameChange}/>
      </div>
    );
  }
});

var VideoBrowser = React.createClass({
  render: function() {
    var job = this.props.job;
    var onSelectedFrameChange = this.props.onSelectedFrameChange;
    var videoNavigators = this.props.videos.map(function(video) {
      return (
        <VideoNavigator job={job}
                        video={video}
                        onSelectedFrameChange={onSelectedFrameChange}
                        key={video['id']}/>
      );
    });

    return (
      <div className="video-browser">
        {videoNavigators}
      </div>
    );
  }
});

var ViewerPanel = React.createClass({
  getInitialState: function() {
    return {
      threshold: 0.3,
      plotType: 'certainty',
    };
  },
  handleThresholdChange: function(e) {
    this.setState({threshold: e.target.value});
  },
  handlePlotTypeChange: function(e) {
    this.setState({plotType: e.target.value});
  },
  handleVideoResize: function(e) {
    this.props.graphics.teardown(this.refs.container);
    var videoElement = ReactDOM.findDOMNode(this.refs.video)
                               .getElementsByTagName('video')[0];
    console.log(videoElement);
    this.props.graphics.setup(this.refs.container, videoElement);
  },
  componentDidMount: function() {
    var frameIndicator = $('<div/>', {'id': 'frame-indicator',
                                      'class': 'timeline-pos-indicator'})
            .css('left', "50%")
            .hide();
    $("#main-panel").append(frameIndicator);

    var classIndicator = $('<div/>', {'id': 'class-indicator',
                                      'class': 'timeline-pos-indicator'})
            .css('left', "50%")
            .hide();
    $("#main-panel").append(classIndicator);

    var videoElement = ReactDOM.findDOMNode(this.refs.video)
                               .getElementsByTagName('video')[0];
    console.log(videoElement);
    this.props.graphics.setup(this.refs.container, videoElement);
    videoElement.addEventListener('resize', this.handleVideoResize);
  },
  componentDidUpdate: function(prevProps, prevState) {
    var videoElement = ReactDOM.findDOMNode(this.refs.video)
                               .getElementsByTagName('video')[0];
    // Setup new graphics if it changes
    if (prevProps.graphics != this.props.graphics) {
      prevProps.graphics.teardown(this.refs.container);
      console.log(videoElement);
      this.props.graphics.setup(this.refs.container, videoElement);
    }
    if (prevProps.video.mediaPath != this.props.video.mediaPath) {
      this.refs.video.load();
    }
    if (this.props.selectedFrame &&
        this.props.selectedFrame.status == 'valid') {
      this.refs.video.seek(this.props.selectedFrame.data.time);
      this.props.graphics.draw(
        videoElement,
        this.props.video,
        this.props.selectedFrame.data);
    }
  },
  render: function() {
    return (
      <div className="viewer-panel" ref="container">
        <Video id="video-viewer" width="100%" ref="video">
          <source src={this.props.video.mediaPath} type="video/mp4" />
        </Video>
        <div className="viewer-controls">
          <select
                  value={this.state.plotType}
                  onChange={this.handlePlotTypeChange}>
            <option value="certainty">Certainty</option>
            <option value="bbox"># of bounding boxes</option>
          </select>
          <div>
            Threshold:
            <input
                    type="number"
                    min="0"
                    max="1"
                    step="any"
                    value={this.state.threshold}
                    onChange={this.handleThresholdChange}/>
          </div>
        </div>
      </div>
    );
  }
});

var VisualizerApp = React.createClass({
  getInitialState: function() {
    return {
      datasets: [{
        id: -1,
        name: "Loading...",
      }],
      selectedDataset: -1,
      videos: [{
        frames: 1,
        width: 1280,
        height: 720,
        id: -1,
        mediaPath: '',
        name: "Loading...",
      }],
      selectedVideo: -1,
      jobs: [{
        id: -1,
        name: "Loading...",
      }],
      selectedJob: -1,
      frameData: [
        [{
          status: 'invalid',
          data: {},
        }]
      ],
      selectedFrame: 0,
      height: 0,
    };
  },
  findDataset: function(id) {
    return _.find(this.state.datasets, function(d) { return d.id == id; });
  },
  findVideo: function(id) {
    return _.find(this.state.videos, function(d) { return d.id == id; });
  },
  findVideoIndex: function(id) {
    return _.findIndex(this.state.videos, function(d) {
      return d.id == id;
    });
  },
  findJob: function(id) {
    return _.find(this.state.jobs, function(d) { return d.id == id; });
  },
  handleSelectedFrameChange: function(d) {
    var frame = d.frame;

    if (this.state.selectedJob != -1) {
      this.loadPredictionData(d.videoId, frame - 1, frame + 1);
    }
    this.setState({
      selectedVideo: d.videoId,
      selectedFrame: d.frame
    });
  },
  handleSelectedDatasetChange: function(datasetId) {
    $.ajax({
      url: "datasets/" + datasetId + "/jobs",
      dataType: "json",
      success: function(jobsData) {
        this.setState({jobs: jobsData});
        //jobMetadata = data[0];
        //jobMetadata.graphics =
        //  graphicsOptions[jobMetadata.featureType];
        //jobMetadata.graphics.setup($("#main-panel"));
      }.bind(this),
    });
    $.ajax({
      url: "datasets/" + datasetId + "/videos",
      dataType: "json",
      success: function(videosData) {
        var frameData = _.map(videosData, function(video) {
          return _.times(video.frames, function(i) {
            return {status: 'invalid', data: {}};
          });
        })
          this.setState({
            selectedVideo: videosData[0].id,
            selectedFrame: 0,
            videos: videosData,
            frameData: frameData
          });
      }.bind(this)
    });
    this.setState({
      selectedDataset: datasetId,
      selectedJob: -1,
    });
  },
  handleSelectedJobChange: function(jobId) {
    var frameData = _.map(this.state.videos, function(video) {
      return _.times(video.frames, function(i) {
        return {status: 'invalid', data: {}};
      });
    })

      this.setState({
        selectedJob: jobId,
        frameData: frameData
      });
  },
  loadDatasetJobData: function(datasetx) {
    $.ajax({
      url: "videos",
      dataType: "json",
      data: {
        job_id: jobsData[0]["id"],
      },
      success: function(videoData) {
        var frameData = _.map(videoData, function(video) {
          return _.times(video.frames, function(i) {
            return {status: 'invalid', data: {}};
          });
        })
          this.setState({
            videos: videoData,
            frameData: frameData
          });
      }.bind(this)
    });
  },
  loadPredictionData: function(videoId, start, end) {
    var frameData = this.state.frameData[videoId];

    var foundStart = false;
    var requestStart = start;
    var requestEnd = end;
    for (var i = start; i < end; ++i) {
      if (frameData[i].status != 'invalid') {
        if (foundStart) {
          requestEnd = i;
          break;
        } else {
          requestStart = i + 1;
        }
      } else {
        foundStart = true;
        frameData =
        update(frameData, {[i]: {status: {$set: 'loading'}}});
      }
    }
    // The entire range is already loaded so we don't need to send a request
    if (requestStart == requestEnd) return;
    $.ajax({
      url: "datasets/" + this.state.selectedDataset +
           "/jobs/" + this.state.selectedJob +
           "/features/" + videoId,
      dataType: "json",
      data: {
        start: requestStart,
        end: requestEnd,
        column: "base_bboxes",
        stride: 1,
        category: -1,
        threshold: $("#threshold-input").val(),
      }
    }).done(function(data) {
      for (var i = 0; i < (requestEnd - requestStart); ++i) {
        var frame = requestStart + i;
        frameData = update(frameData, {
          [frame]: {
            status: {$set: 'valid'},
            data: {$set: data[i]}
          }
        });
      }
      this.setState({
        frameData: update(this.state.frameData,
                          {[videoId]: {$set: frameData}})
      });
    }.bind(this));
  },
  componentDidMount: function() {
    this.setState({height: this._getHeight()});
    window.addEventListener('resize', this._onResize);
    $.ajax({
      url: "datasets",
      dataType: "json",
      success: function(datasetsData) {
        this.setState({datasets: datasetsData});
      }.bind(this),
    });
  },
  _onResize: function() {
    this.setState({height: this._getHeight()});
  },
  _getHeight: function() {
    var top = this.refs.body ? $(this.refs.body).offset().top : 0;
    return window.innerHeight - top;
  },
  render: function() {
    var datasetDivs = _.map(this.state.datasets, function(dataset) {
      return (
        <div className="dataset-info"
             key={dataset.id}
             onClick={()=>this.handleSelectedDatasetChange(dataset.id)}>
          {dataset.name}
        </div>
      );
    }.bind(this));
    var jobDivs = _.map(this.state.jobs, function(job) {
      return (
        <div className="job-info"
             key={job.id}
             onClick={()=>this.handleSelectedJobChange(job.id)}>
          {job.name}
        </div>
      );
    }.bind(this));
    var selectedVideo = this.state.selectedVideo;
    var selectedVideoIndex = this.findVideoIndex(selectedVideo);
    var selectedFrame = this.state.selectedFrame;
    var frame =
    this.state.frameData[selectedVideoIndex][selectedFrame];
    return (
      <div className="visualizer-app">
        <div className="header">
          <h1>Scanner</h1>
          <div className="dataset-panel">
            {datasetDivs}
          </div>
          <div className="job-panel">
            {jobDivs}
          </div>
        </div>
        <div className="body" ref="body" style={{height: this.state.height}}>
          <VideoBrowser job={this.findJob(this.state.selectedJob)}
                        videos={this.state.videos}
                        onSelectedFrameChange={this.handleSelectedFrameChange} />
          <ViewerPanel job={this.findJob(this.state.selectedJob)}
                       video={this.findVideo(selectedVideo)}
                       graphics={detectionGraphics}
                       selectedFrame={frame}/>
        </div>
      </div>
    );
  }
});

$(document).ready(function() {
  ReactDOM.render(<VisualizerApp />, $('#app')[0]);
});
