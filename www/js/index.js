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

var viewWidth = 800;
var viewHeight = 450;

var detectionGraphics = {
    setup: function(mainPanel) {
        mainPanel.append();
        detectionGraphics.svgContainer = d3.select(mainPanel.get(0))
            .append("svg")
            .classed("bbox-container", true)
            .attr("width", viewWidth)
            .attr("height", viewHeight)
            .append("g");
    },
    plotValues: function(sampledData) {
        var certaintyData = [];
        for (var i = 0; i < sampledData.length; ++i) {
            var item = sampledData[i];
            certaintyData.push({
                frame: item.frame,
                value: item.data.certainty,
            });
        }
        return certaintyData;
    },
    show: function() {
    },
    draw: function(videoMetadata, item) {
        var bboxes = detectionGraphics.svgContainer.selectAll(".bbox")
            .data(item.data.bboxes, function (d, i) {
                var id = item.frame + ':' + i;
                return id;
            });
        var w = videoMetadata.width;
        var h = videoMetadata.height;
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

$(document).ready(function() {
    var jobMetadata;
    var videosMetadata;

    $("#threshold-input").val(0.3);

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

    $.ajax({
        url: "jobs",
        dataType: "json",
    }).done(function( data ) {
        jobMetadata = data[0];
        jobMetadata.graphics = graphicsOptions[jobMetadata.featureType];
        jobMetadata.graphics.setup($("#main-panel"));
        $.ajax({
            url: "videos",
            dataType: "json",
            data: {
                job_id: jobMetadata["id"],
            }
        }).done(function( data ) {
            videosMetadata = data;
            for (var i = 0; i < 50 /*videosMetadata.length*/; ++i) {
            //for (var i = 0; i < videosMetadata.length; ++i) {
                setupViewer($("#video-browser"),
                            $("#main-panel"),
                            jobMetadata,
                            videosMetadata[i]);
            }
        });
    });

    // {
    //     'frames': 32000,
    //     'fps': 29.45,
    // };
});

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
    })

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
    var requestRadius = 2;
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

    var labelWidth = 50;
    var labelHeight = 20;

    var axis = $('<div/>', {'class': 'timeline-axis'})
        .css('width', tickWidth * ticks);
    for (var i = 0; i < ticks; ++i) {
        var tick = $('<div/>', {'class': 'timeline-tick'})
            .css('left', tickWidth * i)
            .css('width', tickWidth - 1) // for padding
            .css('top', 0)
            .css('height', tickHeight);
        var tickLabel = $('<div/>', {'class': 'timeline-tick-label'})
            .css('left', tickWidth * i - labelWidth / 2)
            .css('width', labelWidth)
            .css('top', tickHeight)
            .css('height', labelHeight);
        tickLabel.text(Math.round(videoMetadata.frames / ticks * i));
        axis.append(tick);
        axis.append(tickLabel);
    }
    var lastWidth = tickWidth * ticks;
    var tick = $('<div/>', {'class': 'timeline-tick'})
        .css('left', lastWidth - 1)
        .css('width', 0) // for padding
        .css('top', 0)
        .css('height', tickHeight);
    var tickLabel = $('<div/>', {'class': 'timeline-tick-label'})
        .css('left', lastWidth - labelWidth / 2)
        .css('width', labelWidth)
        .css('top', tickHeight)
        .css('height', labelHeight);
    tickLabel.text(videoMetadata.frames);
    axis.append(tick);
    axis.append(tickLabel);

    var selectedFrame = -1;
    var predictionData = _.times(videoMetadata.frames, function (i) {
        return {'status': 'invalid'};
    });

    var handleUpdate = _.debounce(function() {
        if (selectedFrame == -1) return;

        var item = predictionData[selectedFrame];
        var itemData = item.data;
        var status = item.status;
        console.log(selectedFrame);
        console.log(item);
        if (status == 'invalid') {
            loadPredictionData(selectedFrame - requestRadius,
                               selectedFrame + requestRadius);
        } else if (status == 'loading') {
        } else if (status == 'valid') {
            console.log(itemData);
            mainVideo.currentTime = itemData.time;
            jobMetadata.graphics.draw(videoMetadata, itemData);
        }

        if (mainVideoSource.attr("src") != videoMetadata.mediaPath) {
            mainVideoSource.attr("src", videoMetadata.mediaPath);
            mainVideo.load();
            $(mainVideo).one("loadedmetadata", function() {
                handleUpdate();
            });
        }
    }, wait=150);

    function loadPredictionData(start, end) {
        axis.children(".timeline-plot").remove();

        var foundStart = false;
        var requestStart = start;
        var requestEnd = end;
        for (var i = start; i < end; ++i) {
            if (predictionData[i].status == 'loading') {
                if (foundStart) {
                    requestEnd = end;
                    break;
                } else {
                    requestStart = i;
                }
            } else {
                foundStart = true;
                predictionData[i].status = 'loading';
            }
        }
        console.log('range: ' + requestStart + '-' + requestEnd);
        $.ajax({
            url: "jobs/" + jobMetadata["id"] + "/features/" + videoMetadata.id,
            dataType: "json",
            data: {
                start: requestStart,
                end: requestEnd,
                stride: stride,
                category: -1,
                threshold: $("#threshold-input").val(),
            }
        }).done(function(data) {
            for (var i = 0; i < (requestEnd - requestStart); ++i) {
                var frame = requestStart + i;
                predictionData[frame].status = 'valid';
                predictionData[frame].data = data[i];
            }
            handleUpdate();
        });
    }

    // var plotWidth = tickWidth * ticks;
    // setupTimelinePlot(
    //     axis,
    //     mainVideo,
    //     jobMetadata,
    //     videoMetadata,
    //     plotWidth,
    //     tickHeight,
    //     predictionData);

    // User events
    $("#threshold-input").change(function() {
        _.for(predictionData, function(data) {
            data.status = 'invalid';
        });
        handleUpdate();
    });

    axis.mousemove(function(e) {
        var offset = axis.offset();
        var xPos = e.pageX - offset.left;
        var percent = xPos / axis.width();
        var targetedFrame = Math.floor(videoMetadata.frames * percent);

        frameIndicator.text(targetedFrame);
        selectedFrame = targetedFrame;

        handleUpdate();
    });

    axis.mouseenter(function() {
        frameIndicator.show();
        jobMetadata.graphics.show();
    });

    axis.mouseleave(function() {
        frameIndicator.hide();
        jobMetadata.graphics.hide();

        selectedFrame = -1;
    });

    container.append(axis);
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
