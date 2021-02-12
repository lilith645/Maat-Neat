use turtle::Turtle;

pub struct THelper {
  turtle: Turtle,
  width: f64,
  height: f64,
}

impl THelper {
  pub fn new(w: f64, h: f64, speed: i32) -> THelper {
    turtle::start();
    let mut turtle = Turtle::new();
    
    turtle.set_speed("instant");
    turtle.go_to([0.0, 0.0]);
    
    turtle.pen_up();
    
    THelper {
      turtle,
      width: w,
      height: h,
    }
  }
  
  fn distance(&self, x: f64, y: f64, x1: f64, y1: f64) -> f64 {
    let a = x - x1;
    let b = y - y1;
    
    (a*a + b*b).sqrt()
  }
  
  pub fn clear(&mut self) {
    self.turtle.clear();
  }
  
  pub fn draw_dot(&mut self, x: f64, y: f64, turn_degree: f64) {
    self.turtle.go_to([x,y]);
    self.turtle.set_pen_size(10.0);
    self.turtle.pen_down();
    self.turtle.forward(1.0);
    self.turtle.pen_up();
    self.turtle.set_pen_size(1.0);
    /*self.turtle.go_to([x-turn_degree*2.0, y+turn_degree*2.0]);
    self.turtle.pen_down();
    for _ in 0..(360.0/turn_degree) as usize {
      self.turtle.forward(1.0);
      self.turtle.right(turn_degree);
    }
    self.turtle.pen_up();*/
  }
  
  pub fn draw_line(&mut self, x: f64, y: f64, x1: f64, y1: f64) {
    self.turtle.go_to([x, y]);
    
    self.turtle.pen_down();
    self.turtle.turn_towards([x1, y1]);
    self.turtle.forward(self.distance(x,y,x1,y1));
    self.turtle.pen_up();
  }
  
  pub fn pen_size(&mut self, size: f64) {
    self.turtle.set_pen_size(size);
  }
  
  pub fn set_colour(&mut self, c: &str) {
    self.turtle.set_pen_color(c);
  }
  
  pub fn wait(&mut self) {
    self.turtle.wait_for_click();
  }
}
