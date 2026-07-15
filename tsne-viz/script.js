const w = window.innerWidth;
const h = window.innerHeight;
const padding = 200;
const imgSize = 25;

const svg = d3.select("#viz");
const g = svg.append("g");

const zoom = d3.zoom()
  .scaleExtent([0.3, 10])
  .on("zoom", e => g.attr("transform", e.transform));
svg.call(zoom);

const tooltip = d3.select("#tooltip");

d3.json("tsne_data.json").then(data => {
  const xScale = d3.scaleLinear()
    .domain(d3.extent(data, d => d.x))
    .range([padding, w - padding]);

  const yScale = d3.scaleLinear()
    .domain(d3.extent(data, d => d.y))
    .range([padding, h - padding]);

  const nodes = data.map(d => ({
    ...d,
    px: xScale(d.x),
    py: yScale(d.y)
  }));

  const sim = d3.forceSimulation(nodes)
    .force("x", d3.forceX(d => d.px).strength(0.9))
    .force("y", d3.forceY(d => d.py).strength(0.9))
    .force("collide", d3.forceCollide(12))
    .stop();

  for (let i = 0; i < 300; i++) sim.tick();

  g.selectAll("image")
    .data(nodes)
    .enter()
    .append("image")
    .attr("href", d => d.img)
    .attr("x", d => d.x - imgSize / 2)
    .attr("y", d => d.y - imgSize / 2)
    .attr("width", imgSize)
    .attr("height", imgSize)
    .style("cursor", "pointer")
    .on("mouseover", (e, d) => {
      tooltip.style("display", "block")
        .html(`<b>${d.name}</b><br>Class: ${d.cls}<br>ID: ${d.id}`);
    })
    .on("mousemove", e => {
      tooltip
        .style("left", (e.pageX + 12) + "px")
        .style("top", (e.pageY - 20) + "px");
    })
    .on("mouseout", () => tooltip.style("display", "none"));
});